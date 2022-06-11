# stream audio to vosk server
import asyncio
import websockets
import sounddevice as sd

import config


def callback():
    # TODO push audio block to queue
    pass


def open_device(device_num):
    print(sd.query_devices())
    samplerate = 80
    blocksize = 8000
    return sd.RawInputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        device=device_num,
        dtype='int16',
        channels=1,
        callback=callback
    )


async def run(uri):
    audio_queue = asyncio.Queue()
    with open_device(1) as device:
        async with websockets.connect(uri) as sock:
            await sock.connect('{ "config": {"sample_rate": %d }' % (device.samplerate))

            while True:
                data = await audio_queue.get()
                await sock.send(data)
                await sock.recv()

            await sock.send('{"eof": 1}')
            await sock.recv()

async def test():
    async with websockets.connect(config.VOSK_URL) as sock:
        pass

if __name__ == "__main__":
    asyncio.run(test())
    # asyncio.run(run(config.VOSK_URL))

