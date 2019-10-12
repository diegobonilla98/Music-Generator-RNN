import librosa
import numpy as np
from pysndfx import AudioEffectsChain


def pianoSynte(notes):
    path = "piano notes/"
    # librosa.output.write_wav('song_generated.wav', X, sr)

    A3, _ = librosa.load(path + "A3.wav")
    A4, _ = librosa.load(path + "A4.wav")
    Ab3, _ = librosa.load(path + "Ab3.wav")
    Ab4, _ = librosa.load(path + "Ab4.wav")
    B3, _ = librosa.load(path + "B3.wav")
    B4, _ = librosa.load(path + "B4.wav")
    Bb3, _ = librosa.load(path + "Bb3.wav")
    Bb4, _ = librosa.load(path + "Bb4.wav")
    C3, _ = librosa.load(path + "C3.wav")
    C4, _ = librosa.load(path + "C4.wav")
    D3, _ = librosa.load(path + "D3.wav")
    D4, _ = librosa.load(path + "D4.wav")
    Db3, _ = librosa.load(path + "Db3.wav")
    Db4, _ = librosa.load(path + "Db4.wav")
    E3, _ = librosa.load(path + "E3.wav")
    E4, _ = librosa.load(path + "E4.wav")
    Eb3, _ = librosa.load(path + "Eb3.wav")
    Eb4, _ = librosa.load(path + "Eb4.wav")
    F3, _ = librosa.load(path + "F3.wav")
    F4, _ = librosa.load(path + "F4.wav")
    G3, _ = librosa.load(path + "G3.wav")
    G4, _ = librosa.load(path + "G4.wav")
    Gb3, _ = librosa.load(path + "Gb3.wav")
    Gb4, _ = librosa.load(path + "Gb4.wav")
    silence, _ = librosa.load(path + "silence.wav")

    # X = A3, _ = librosa.load(path + "A3.wav")
    # X = list(A4) + list(Ab3) + list(Ab4) + list(B3) + list(B4) + list(Bb3) + list(Bb4) + list(C3) + list(C4) + \
    #     list(D3) + list(D4) + list(Db3) + list(Db4) + list(E3) + list(E4) + list(Eb3) + list(Eb4) + list(F3) + \
    #     list(F4) + list(G3) + list(G4) + list(Gb3) + list(Gb4)
    #
    # X = np.array(X)
    # librosa.output.write_wav('song_generated.wav', X, 22050)

    notes_sound = {
        0: silence,
        1: C3,
        2: Db3,
        3: D3,
        4: Eb3,
        5: E3,
        6: F3,
        7: Gb3,
        8: G3,
        9: Ab3,
        10: A3,
        11: Bb3,
        12: B3,
        13: C4,
        14: Db4,
        15: D4,
        16: Eb4,
        17: E4,
        18: F4,
        19: Gb4,
        20: G4,
        21: Ab4,
        22: A4,
        23: Bb4,
        24: B4
    }
    X = []
    for note in notes:
        X += list(notes_sound[note])
    X = np.array(X)

    fx = (
        AudioEffectsChain()
        .reverb(reverberance=40, room_scale=60)
        .normalize()
    )

    X = fx(X)

    librosa.output.write_wav('song_generated_piano.wav', X, 22050)
