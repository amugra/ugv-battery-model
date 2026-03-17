#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "pico/stdlib.h"
#include "pico/bootrom.h"
#include "pico/time.h"

#include "hardware/adc.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"

#include "tusb.h"

#define LOAD_GPIO               17
#define TRIGGER_GPIO            15
#define ADC_GPIO                26
#define ADC_INPUT               0

#define SAMPLE_RATE_HZ          350000u
#define BUF_SAMPLES             4096u

#define LOAD_ON_TIME_US         (30000000ull)   
#define LOAD_OFF_TIME_US        (120000000ull) 
#define BASELINE_DELAY_US       (100000ull)   

#define LOAD_ON_LEVEL           0
#define LOAD_OFF_LEVEL          1

// ADC / DMA buffers
static uint16_t adc_buf_a[BUF_SAMPLES];
static uint16_t adc_buf_b[BUF_SAMPLES];

static volatile bool buf_a_ready = false;
static volatile bool buf_b_ready = false;
static volatile bool dma_using_a = true;

static int dma_chan;
static dma_channel_config dma_cfg;
static bool adc_dma_running = false;

// Run state
typedef enum {
    RUN_STATE_IDLE = 0,
    RUN_STATE_BASELINE,
    RUN_STATE_ON,
    RUN_STATE_OFF
} run_state_t;

static volatile run_state_t run_state = RUN_STATE_IDLE;
static uint64_t next_transition_us = 0;

// Event packet
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];        // 0xEE, 0xEE
    uint8_t  type;            // 1 = run state change
    uint8_t  reserved;
    uint64_t timestamp_us;
    uint32_t value;
} event_packet_t;

static inline void load_set(bool on)
{
    gpio_put(LOAD_GPIO, on ? LOAD_ON_LEVEL : LOAD_OFF_LEVEL);
}

static void usb_write_all_blocking(const uint8_t *data, uint32_t len)
{
    uint32_t sent = 0;

    while (sent < len) {
        tud_task();

        if (!tud_cdc_connected()) {
            return;
        }

        uint32_t avail = tud_cdc_write_available();
        if (avail == 0) {
            continue;
        }

        uint32_t chunk = len - sent;
        if (chunk > avail) {
            chunk = avail;
        }

        uint32_t written = tud_cdc_write(data + sent, chunk);
        sent += written;
        tud_cdc_write_flush();
    }
}

static void usb_write_str(const char *s)
{
    if (!tud_cdc_connected()) {
        return;
    }

    usb_write_all_blocking((const uint8_t *)s, (uint32_t)strlen(s));
}

static void send_event(uint8_t type, uint32_t value, uint64_t timestamp_us)
{
    if (!tud_cdc_connected()) {
        return;
    }

    event_packet_t evt = {
        .magic = {0xEE, 0xEE},
        .type = type,
        .reserved = 0,
        .timestamp_us = timestamp_us,
        .value = value
    };

    usb_write_all_blocking((const uint8_t *)&evt, sizeof(evt));
}

// DMA IRQ
static void dma_handler(void)
{
    dma_hw->ints0 = 1u << dma_chan;

    if (dma_using_a) {
        buf_a_ready = true;
        dma_using_a = false;
        dma_channel_set_write_addr(dma_chan, adc_buf_b, true);
    } else {
        buf_b_ready = true;
        dma_using_a = true;
        dma_channel_set_write_addr(dma_chan, adc_buf_a, true);
    }
}

// ADC / DMA control
static void adc_dma_init_once(void)
{
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;

    adc_init();
    adc_gpio_init(ADC_GPIO);
    adc_select_input(ADC_INPUT);

    adc_fifo_setup(
        true,    // en
        true,    // dreq_en
        1,       // dreq_thresh
        false,   // err_in_fifo
        false    // byte_shift
    );

    adc_set_clkdiv(59999.0f);

    dma_chan = dma_claim_unused_channel(true);
    dma_cfg = dma_channel_get_default_config(dma_chan);

    channel_config_set_transfer_data_size(&dma_cfg, DMA_SIZE_16);
    channel_config_set_read_increment(&dma_cfg, false);
    channel_config_set_write_increment(&dma_cfg, true);
    channel_config_set_dreq(&dma_cfg, DREQ_ADC);

    dma_channel_configure(
        dma_chan,
        &dma_cfg,
        adc_buf_a,
        &adc_hw->fifo,
        BUF_SAMPLES,
        false
    );

    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
    irq_set_enabled(DMA_IRQ_0, true);
}

static void adc_dma_start(void)
{
    if (adc_dma_running) {
        return;
    }

    adc_dma_init_once();

    buf_a_ready = false;
    buf_b_ready = false;
    dma_using_a = true;

    adc_fifo_drain();
    adc_run(false);

    dma_channel_abort(dma_chan);
    dma_channel_set_write_addr(dma_chan, adc_buf_a, false);
    dma_channel_set_read_addr(dma_chan, &adc_hw->fifo, false);
    dma_channel_set_trans_count(dma_chan, BUF_SAMPLES, false);

    adc_run(true);
    dma_channel_start(dma_chan);

    adc_dma_running = true;
}

static void adc_dma_stop(void)
{
    if (!adc_dma_running) {
        return;
    }

    adc_run(false);
    dma_channel_abort(dma_chan);
    adc_fifo_drain();

    adc_dma_running = false;
    buf_a_ready = false;
    buf_b_ready = false;
}

// Run control
static void start_run(void)
{
    uint64_t now = time_us_64();

    load_set(false);          
    adc_dma_start();

    run_state = RUN_STATE_BASELINE;
    next_transition_us = now + BASELINE_DELAY_US;

    // Binary marker in stream
    send_event(1, (uint32_t)RUN_STATE_BASELINE, now);
}

static void stop_run(void)
{
    uint64_t now = time_us_64();

    load_set(false);
    adc_dma_stop();

    run_state = RUN_STATE_IDLE;
    next_transition_us = 0;

    // This is sent before stopping if USB is still alive enough
    send_event(1, (uint32_t)RUN_STATE_IDLE, now);
}

static void service_run_state_machine(uint64_t now_us)
{
    if (run_state == RUN_STATE_IDLE) {
        return;
    }

    if (now_us < next_transition_us) {
        return;
    }

    switch (run_state) {
        case RUN_STATE_BASELINE:
            load_set(true);
            run_state = RUN_STATE_ON;
            next_transition_us = now_us + LOAD_ON_TIME_US;
            send_event(1, (uint32_t)RUN_STATE_ON, now_us);
            break;

        case RUN_STATE_ON:
            load_set(false);
            run_state = RUN_STATE_OFF;
            next_transition_us = now_us + LOAD_OFF_TIME_US;
            send_event(1, (uint32_t)RUN_STATE_OFF, now_us);
            break;

        case RUN_STATE_OFF:
            load_set(true);
            run_state = RUN_STATE_ON;
            next_transition_us = now_us + LOAD_ON_TIME_US;
            send_event(1, (uint32_t)RUN_STATE_ON, now_us);
            break;

        default:
            break;
    }
}

// Command parser
static void print_status(void)
{
    // Only safe-ish to print text while idle, otherwise it pollutes the stream.
    if (run_state == RUN_STATE_IDLE) {
        usb_write_str("status: idle\r\n");
    } else {
        usb_write_str("status: running\r\n");
    }
}

static void service_commands(void)
{
    if ((gpio_get(TRIGGER_GPIO)) && (run_state == RUN_STATE_IDLE)) {
        start_run();
        }

    while (tud_cdc_available()) {
        char c = tud_cdc_read_char();

        switch (c) {
            case 'x':
            case 'X':
                stop_run();
                break;

            case '?':
                print_status();
                break;

            case 'b':
            case 'B':
                usb_write_str("rebooting to bootsel\r\n");
                sleep_ms(50);
                reset_usb_boot(0, 0);
                break;

            default:
                break;
        }
    }
}

int main(void)
{
    gpio_init(LOAD_GPIO);
    gpio_set_dir(LOAD_GPIO, GPIO_OUT);
    gpio_init(TRIGGER_GPIO);
    gpio_set_dir(TRIGGER_GPIO, GPIO_IN);
    load_set(false);   

    tusb_init();

    sleep_ms(250);

    while (1) {
        tud_task();

        service_commands();
        service_run_state_machine(time_us_64());

        if (adc_dma_running) {
            if (buf_a_ready) {
                buf_a_ready = false;
                usb_write_all_blocking((const uint8_t *)adc_buf_a, sizeof(adc_buf_a));
            }

            if (buf_b_ready) {
                buf_b_ready = false;
                usb_write_all_blocking((const uint8_t *)adc_buf_b, sizeof(adc_buf_b));
            }
        }
    }
}
