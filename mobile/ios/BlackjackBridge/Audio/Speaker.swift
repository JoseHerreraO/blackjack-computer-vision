//
// Speaker.swift
//
// Speaks the PC's "say" commands. The Meta glasses are a Bluetooth audio device of the
// iPhone, so with a playback audio session the speech comes out of the glasses' speakers
// whenever they are connected (otherwise out of the phone). The output route is logged with
// every phrase so it is easy to check where it went.
//
// A new phrase interrupts the one being spoken: the PC sends the current advice, and an old
// one is no longer useful.
//

import AVFoundation
import Foundation
import Observation

@MainActor
@Observable
final class Speaker: NSObject {
    static let defaultLanguage = "es-ES"

    private(set) var isSpeaking = false
    /// Current audio output, e.g. "Ray-Ban Meta (BluetoothA2DPOutput)".
    private(set) var outputRoute = "–"

    @ObservationIgnored private let synthesizer = AVSpeechSynthesizer()
    @ObservationIgnored private let log: LogStore
    @ObservationIgnored private var sessionReady = false

    init(log: LogStore) {
        self.log = log
        super.init()
        synthesizer.delegate = self
    }

    /// Speaks `text` in `language` (BCP 47, e.g. "es-ES"), interrupting any phrase in progress.
    func say(_ text: String, language: String?) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        prepareSession()

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let language = (language?.isEmpty == false ? language : nil) ?? Self.defaultLanguage
        let utterance = AVSpeechUtterance(string: text)
        if let voice = AVSpeechSynthesisVoice(language: language) {
            utterance.voice = voice
        } else {
            log.add("No voice for \(language), using the default one")
        }
        synthesizer.speak(utterance)
        log.add("Saying \"\(text)\" (\(language)) on \(outputRoute)")
    }

    /// Plays a short phrase to check the sound reaches the glasses without the PC.
    func test() {
        say("Prueba de voz. Pide carta.", language: Self.defaultLanguage)
    }

    private func prepareSession() {
        let session = AVAudioSession.sharedInstance()
        if !sessionReady {
            do {
                // Playback goes to the active Bluetooth (A2DP) output, i.e. the glasses when
                // they are connected. Other audio is lowered while speaking.
                try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
                try session.setActive(true)
                sessionReady = true
            } catch {
                log.add("Audio session error: \(error.localizedDescription)")
            }
        }
        let outputs = session.currentRoute.outputs.map { "\($0.portName) (\($0.portType.rawValue))" }
        outputRoute = outputs.isEmpty ? "no output" : outputs.joined(separator: ", ")
    }
}

extension Speaker: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = true }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = self.synthesizer.isSpeaking }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = self.synthesizer.isSpeaking }
    }
}
