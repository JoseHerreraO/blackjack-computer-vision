# 01 Phone connection, glasses → iPhone → PC

The camera is on the player's head, in a pair of Ray-Ban Meta glasses. The glasses cannot talk to
the PC directly, so an iPhone app, **BJ Bridge** (`mobile/ios/`), receives their video with Meta's
Wearables Device Access Toolkit (DAT 1.0.1) and forwards every frame to the PC over Wi-Fi. The
phone does no vision at all. It is only a bridge, and it also speaks the PC's advice through the
glasses' speakers.

```
Ray-Ban Meta ──Bluetooth (DAT)──► iPhone, BJ Bridge ──TCP 5050, Wi-Fi──► PC
                                         ▲                                  │
                                         └──────── COMMAND say / status ────┘
```

## Building an iOS app without a Mac

Nobody in the team has a Mac, and an iOS app has to be compiled with Xcode. The way around it is
GitHub Actions, which offers macOS machines.

- No Xcode project is stored in the repository. `mobile/ios/project.yml` describes it and
  **XcodeGen** generates the project and its `Info.plist` on the runner.
- `.github/workflows/ios.yml` runs on every push that touches `mobile/ios/` (and by hand from the
  Actions tab). It uses a `macos-26` runner, because DAT 1.0.1 needs Xcode 26.4 or newer, and
  builds the app **unsigned**.
- The result is the artifact `BlackjackBridge-<run>`, a `.ipa` file kept for 14 days.
- macOS minutes are free because the repository is public. While it was private the jobs often
  failed with "The job was not acquired by Runner".

The unsigned `.ipa` is installed from Windows with **Sideloadly** and a free Apple ID, which signs
it during the install. iTunes and iCloud have to be the versions from Apple's website, not the
Microsoft Store ones, or Sideloadly does not see the iPhone. On the phone, Developer Mode must be
on and the developer certificate trusted once. A free signature lasts **7 days**, so the app has
to be installed again every week.

## Getting the glasses' video

1. In the Meta AI app, Developer Mode is enabled by tapping the version number five times in the
   app information (this is not the iPhone's Developer Mode).
2. BJ Bridge registers with Meta AI (*Connect glasses*), which opens Meta AI to approve the app.
3. *Start session* opens a session with the glasses and *Start stream* starts the camera. The first
   time Meta AI asks for camera permission.

The `Info.plist` must declare the `external-accessory` background mode and the
`com.meta.ar.wearable` accessory protocol, because DAT streams over Bluetooth Classic. Meta's own
sample app does not declare them, and without them the stream fails with "device not connected".
That was the first problem found in the feasibility test.

DAT offers 720×1280, 504×896 and 360×640 at 2 to 30 fps and lowers them by itself when Bluetooth
cannot keep up. Asking for 24 fps at 720×1280 still delivered 15 fps, so **15 fps is the real
ceiling**. The three resolutions show the same field of view (feature matching between them gave a
scale of 0.98 to 0.99), so a higher resolution only adds detail.

## The network

The PC is a TCP server on port 5050 and the iPhone connects to it. Two things had to be learned.

The **university Wi-Fi isolates its clients**, so the phone and the PC cannot see each other and
the app only reports `PC not reachable (POSIXErrorCode 60)`. The iPhone's own hotspot was not an
option either (the carrier blocks it). The solution is the **Windows mobile hotspot** (*Zona con
cobertura inalámbrica móvil*). The PC becomes the access point at `192.168.137.1`, the iPhone
joins it, and the app connects to `192.168.137.1:5050`.

The hotspot must be on the **5 GHz band**. On 2.4 GHz it competes with the glasses' Bluetooth, the
SDK lowers the stream to 360×640 and the iPhone eventually drops off the Wi-Fi. On 5 GHz the
stream held **720×1280 at a steady 15.0 fps**.

## The protocol

Every message is `[1 byte type][uint32 big-endian payload length][payload]`. It is implemented in
`mobile/ios/BlackjackBridge/Networking/BridgeProtocol.swift` and on the PC in
`src/blackjack_vision/capture/bridge.py` (and the test receiver `mobile/ios/tools/pc_receiver.py`).

| Type | Direction | Payload |
|---|---|---|
| `0x01` HELLO | iPhone → PC | JSON with the device, width, height, fps, JPEG quality and app version. Sent on connect and when the resolution or fps change |
| `0x02` FRAME | iPhone → PC | Capture time (int64 ms since epoch) and the JPEG (quality 0.8), not rotated, cropped or scaled |
| `0x03` LOG | iPhone → PC | One line of the app's log, the only console we have without a Mac |
| `0x10` COMMAND | PC → iPhone | `{"type":"say","text":...,"lang":"es-ES"}` or `{"type":"status","text":...}` |

Frames are **latest only**. The phone keeps at most one frame in flight and replaces any newer one
that is waiting, so a slow PC never builds up delay. On the PC `BridgeSource` does the same, so a
slow pipeline always works on the newest frame. If the iPhone disappears without closing the
connection (Wi-Fi lost, app in the background), a new connection replaces the dead one and TCP
keepalive detects the lost peer in about 15 s. Before that fix the receiver waited forever on the
dead socket.

## Voice through the glasses

A `say` command is spoken with `AVSpeechSynthesizer` on a playback audio session, so the speech
goes to the active Bluetooth output, which is the glasses when they are connected. A new phrase
interrupts the previous one, because only the latest advice matters. The app shows the audio output
and has a *Test voice* button. A `status` command only updates a line on the phone's screen. The
PC sends `say` only when the advice changes and when a round ends, never every frame. The voice
still has to be tested live with the video stream running (#22).

## Tools

`mobile/ios/tools/pc_receiver.py` is a small server to check the bridge without the full pipeline.
It shows the video with fps and latency, prints HELLO and LOG, sends `say` and `status` typed in the
terminal, and saves frames untouched (`s` one, `r` record). All the recordings used in this project
were made with it.

See also `docs/ios_bridge.md` for the full troubleshooting table.
