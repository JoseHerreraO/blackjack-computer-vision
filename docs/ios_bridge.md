# iOS bridge: Meta glasses → iPhone → PC without a Mac

The glasses' camera reaches the PC through a small iPhone app, **BJ Bridge** (`mobile/ios/`).
The app receives the video from the Ray-Ban Meta glasses with Meta's Wearables Device Access
Toolkit (DAT) and sends every frame to the PC over TCP. The PC can answer with commands that the
app speaks through the glasses (`say`) or shows on screen (`status`).

```
Ray-Ban Meta --Bluetooth--> iPhone (BJ Bridge, DAT SDK) --TCP 5050, Wi-Fi--> PC (BridgeSource / pc_receiver.py)
                                    ^                                        |
                                    +---------------- COMMAND (say/status) --+
```

Nobody on the team has a Mac, so the app is built in the cloud and installed from Windows.

## 1. How the app is built (GitHub Actions)

- No Xcode project is versioned. `mobile/ios/project.yml` describes it and **XcodeGen** generates
  `BlackjackBridge.xcodeproj` and `Generated/Info.plist` on the runner.
- `.github/workflows/ios.yml` runs on every push that touches `mobile/ios/**` (and manually from
  the Actions tab). It uses a `macos-26` runner, because DAT 1.0.1 needs Xcode 26.4 or newer, and
  builds the app **unsigned** (`CODE_SIGNING_ALLOWED=NO`).
- The result is the artifact **`BlackjackBridge-<run>`** with `BlackjackBridge-<run>-<sha>.ipa`,
  the build log and the generated `Info.plist`. Artifacts are kept for 14 days.
- The repo is public, so macOS minutes are free. While it was private, macOS jobs often failed
  with "The job was not acquired by Runner". If that happens, re-run the job.
- The build checks that the `Info.plist` keeps the keys that step 0 proved necessary
  (`external-accessory`, `com.meta.ar.wearable`, `NSLocalNetworkUsageDescription`).

The same workflow can build Meta's official CameraAccess sample (`target=sample`), which we used
to check feasibility and measure frame rates before writing our app.

## 2. Installing it on the iPhone (Sideloadly, free Apple ID)

Once on the PC:

1. Install **iTunes** and **iCloud** from Apple's website, not from the Microsoft Store, otherwise
   Sideloadly does not see the iPhone. The links are on sideloadly.io. They only need to be
   installed, not used.
2. Install **Sideloadly** from sideloadly.io.
3. Connect the iPhone with a data cable, unlock it and tap **Trust this computer**.

Every time there is a new build:

1. In GitHub, open Actions → iOS → the last green run → Summary → Artifacts and download
   `BlackjackBridge-<run>`. Unzip it.
2. Drag the `.ipa` onto Sideloadly, check the Apple ID and press **Start**. The Apple ID password
   and verification code go only into Sideloadly. It ends with "Done".

Once on the iPhone (after the first install):

1. Settings → Privacy & Security → **Developer Mode** → on (the iPhone restarts).
2. Settings → General → VPN & Device Management → the Apple ID → **Trust**.

With a free Apple ID the signature **expires after 7 days** and the app has to be installed
again (same steps, it replaces the old one). Limits are 3 sideloaded apps at a time and 10 new
App IDs per week, so delete old test apps. The app needs no paid entitlements.

## 3. Meta AI setup (once)

- DAT needs Meta AI 292+ and glasses firmware 128+ (tested with Meta AI 299.0.0, firmware 130).
- Enable Meta AI's own developer mode: Meta AI → Settings → App info → tap the **version number
  five times** → Developer Mode → Enable. This is not the iPhone's Developer Mode.
- In BJ Bridge tap **Connect glasses (Meta AI)**. Meta AI opens to approve the app and comes
  back. Approved apps are listed in Meta AI → Settings → Apps (App connections).
- The first **Start stream** asks for camera permission in Meta AI. Choose "Allow always".

## 4. Network

The iPhone and the PC must see each other on the same network.

- **The university Wi-Fi does not work.** It isolates clients, so the app only shows
  `PC not reachable (... POSIXErrorCode 60 ...)`, a timeout.
- Use the **Windows mobile hotspot** (Settings → Network & Internet → Mobile hotspot) and connect
  the iPhone to it. The PC is then **`192.168.137.1`**, so the app address is
  `192.168.137.1:5050`.
- Set the hotspot to the **5 GHz band** (Mobile hotspot → Edit → Network band). On 2.4 GHz it
  competes with the glasses' Bluetooth: the SDK lowers the resolution and the iPhone eventually
  drops off the Wi-Fi.
- A home Wi-Fi without client isolation also works. Windows may ask to allow Python through the
  firewall the first time. Allow it.
- iOS asks for **Local Network** permission the first time the app connects. If it was denied,
  enable it in Settings → Privacy & Security → Local Network → BJ Bridge.

## 5. Using it

On the PC, start a receiver first:

```bash
# Test receiver: shows the video, prints HELLO/LOG, s = save frame, r = record, q = quit
python mobile/ios/tools/pc_receiver.py            # --every 5 keeps 1 frame out of 5
# Or the pipeline source (see docs/card_detection.md once merged)
python scripts/detect_cards.py --source bridge
```

`pc_receiver.py` prints every address of the PC and saves frames, untouched, to
`data/samples/<date-time>/` (ignored by git). The keys only work with the video window focused.

In the app:

1. Type the PC address (`192.168.137.1:5050`) and tap **Connect**. Status → PC becomes
   `connected`. It retries every 2 s until you tap Disconnect.
2. With the glasses on, tap **Start session**, then **Start stream**.
3. The video shows in the app and on the PC. The log panel (Copy button) is the only console we
   have, and every line is also sent to the PC.

The address field can only be edited while disconnected.

### Voice

A `say` command is spoken with `AVSpeechSynthesizer` in the language it gives (`es-ES` by
default). The audio session is `.playback` with the `.spokenAudio` mode, so the speech goes to the
active Bluetooth output, which is the glasses' speakers whenever they are connected, and other
audio is lowered while it plays. A new phrase interrupts the one being spoken, because only the
latest advice matters. Every phrase is logged together with the output it went to, and the
**Voice** section shows the current output with a **Test voice** button that works without the
PC. A `status` command only updates **Status → PC status**.

From the PC: type `say pide carta` in the terminal of `pc_receiver.py`, or call
`BridgeSource.say("Pide carta")` in the pipeline.

## 6. Protocol (iPhone ↔ PC contract)

The PC is the TCP server on port **5050** and the iPhone is the client. Every message is
`[1 byte type][uint32 big-endian payload length][payload]`. Implemented in
`mobile/ios/BlackjackBridge/Networking/BridgeProtocol.swift` and on the PC side in
`mobile/ios/tools/pc_receiver.py` and `src/blackjack_vision/capture/bridge.py`. Do not change one
side without the other.

| Type | Direction | Payload |
|------|-----------|---------|
| `0x01` HELLO | iPhone → PC | JSON `{"device","width","height","fps","jpeg_quality","app_version"}`, sent on connect and whenever the resolution or fps changes |
| `0x02` FRAME | iPhone → PC | int64 big-endian capture time (ms since epoch) + JPEG (quality 0.8), not rotated, cropped or scaled |
| `0x03` LOG | iPhone → PC | UTF-8 text, one log line |
| `0x10` COMMAND | PC → iPhone | JSON `{"type":"say","text":"...","lang":"es-ES"}` or `{"type":"status","text":"..."}` |

Frames are **latest only**. The app keeps at most one frame in flight and replaces any newer one
that is waiting, so old frames are dropped instead of queued and the PC never falls behind. DAT
gives no wall-clock capture time, so the timestamp is the arrival time on the iPhone.

## 7. Stream settings

DAT offers 720×1280, 504×896 and 360×640 at 2, 7, 15, 24 or 30 fps, and lowers them by itself
when Bluetooth cannot keep up (the app logs every size change and re-sends HELLO).

- With the official sample, 720×1280 at 15 fps gave exactly 15 fps. Asking for 24 fps still gave
  15 fps.
- On a 2.4 GHz hotspot the SDK dropped to 360×640. On the **5 GHz** hotspot **720×1280 @ 15 fps**
  holds (28 s at a steady 15.0 fps on 2026-11-18), and that is the resolution to use: the three
  resolutions have the same field of view, and the cards need the pixels (at 360×640 a third of
  them are never recognised). Choose it in the app from the start so the frame size does not
  change mid-stream.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Preview or stream fails with "device not connected" | The `Info.plist` lacks `external-accessory` / `com.meta.ar.wearable`. Ours has them. The official sample does not. |
| "Glasses: not found", Start session disabled | Put the glasses on (or open the temples) with Bluetooth on. |
| Sideloadly "Login failed (996996)", `Fake2 Anisette` in the log | Open iTunes, sign in once, restart Sideloadly. |
| Sideloadly "(1100) Your session has expired" | Delete and retype the Apple ID so it asks for the password and code again. |
| `PC not reachable ... POSIXErrorCode 60` | Different network or client isolation. Use the 5 GHz Windows hotspot and `192.168.137.1:5050`. |
| Status stays `disconnected` after Connect | The address is invalid (empty, spaces or the grey placeholder). |
| Receiver prints "message too large (1163141167 bytes)" | Something else (a browser) opened the port. The address goes in the app, not in a browser. |
| Resolution drops to 504×896 or 360×640, then the connection dies | Hotspot on 2.4 GHz. Switch to 5 GHz. |
| The app logs "TCP client is not implemented yet" | An old build is installed. Check Status → App version. |
| `say` is heard from the iPhone, not the glasses | The glasses are not the Bluetooth audio output. Check **Voice → Output** (it should name the glasses) and pick them in Control Center → AirPlay. |
| Nothing is heard | Ring/silent switch or volume. Try **Test voice**. The log says which output each phrase went to. |
| The app stops opening after a week | The free Apple ID signature expired (7 days). Install the last `.ipa` again with Sideloadly. |

## 9. Status

- Steps 0-4 done: feasibility, CI build, glasses video on screen, TCP client (HELLO, FRAME,
  LOG, COMMAND) and `say` spoken through the glasses.
- Since 0.2.0 the app reports its real version, `0.2.0 (<CI run number>)`.
