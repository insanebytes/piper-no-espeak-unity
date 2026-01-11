using System;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;

namespace PiperTTS
{
    public static class NumberExtensions
    {
        public static string ToWords(this int number) => NumberToText.Convert(number);
        public static string ToWords(this long number) => NumberToText.Convert(number);
        public static string ToWords(this double number) => NumberToText.Convert(number);
    }

    public class PiperTTS : MonoBehaviour
    {
        [Header("Piper")]
        public string piperModelPath = string.Empty;
        public string piperConfigPath = string.Empty;

        [Header("Phonemizer")]
        public string phonemizerModelPath = string.Empty;
        public string phonemizerConfigPath = string.Empty;
        public string phonemizerDictPath = string.Empty;


        [Header("Config")]
        [Range(0.0f, 1.0f)]
        public float commaDelay = 0.1f;

        [Range(0.0f, 1.0f)]
        public float periodDelay = 0.5f;

        [Range(0.0f, 1.0f)]
        public float questionExclamationDelay = 0.6f;

        protected PiperModel piper;
        protected PhonemizerModel phonemizer;
        protected AudioSource audioSource;

        public delegate void StatusChangedDelegate(ModelStatus status);
        public event StatusChangedDelegate OnStatusChanged;

        private ModelStatus _status = ModelStatus.Init;

        // Public getter, no public setter
        public ModelStatus status
        {
            get => _status;
            protected set
            {
                if (_status != value)
                {
                    _status = value;
                    OnStatusChanged?.Invoke(_status);
                }
            }
        }

        public void InitModel()
        {
            if (string.IsNullOrEmpty(piperModelPath) || string.IsNullOrEmpty(piperConfigPath))
            {
                return;
            }

            if (string.IsNullOrEmpty(phonemizerModelPath) || string.IsNullOrEmpty(phonemizerConfigPath))
            {
                return;
            }

            if (_status != ModelStatus.Init)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Loading;
            StartCoroutine(RunInitModel());
        }

        IEnumerator RunInitModel()
        {
            Debug.Log($"Load piper tts model");

            piper.modelPath = piperModelPath;
            piper.configPath = piperConfigPath;
            piper.InitModel();

            phonemizer.modelPath = phonemizerModelPath;
            phonemizer.configPath = phonemizerConfigPath;
            phonemizer.dictPath = phonemizerDictPath;
            phonemizer.InitModel();

            yield return new WaitWhile(() => phonemizer.status != ModelStatus.Ready);
            yield return new WaitWhile(() => piper.status != ModelStatus.Ready);

            Debug.Log("Load model done");

            status = ModelStatus.Ready;
        }

        public void Prompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                return;
            }

            if (status != ModelStatus.Ready)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            StartCoroutine(SynthesizeAndPlay(prompt));
        }

        string PreProcessNumbers(string text)
        {
            // Regex to find numbers (including decimals)
            return Regex.Replace(text, @"\d+(\.\d+)?", match =>
            {
                try
                {
                    if (double.TryParse(match.Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double number))
                    {
                        return number.ToWords();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning(e.Message);
                }
                return match.Value;
            });
        }

        IEnumerator SynthesizeAndPlay(string prompt)
        {
            string text = PreProcessNumbers(prompt);
            string delayPunctuationPattern = @"([,.?!;:])";
            string nonDelayPunctuationPattern = @"[^\w\s,.?!;:]";

            string[] parts = Regex.Split(text, delayPunctuationPattern);

            foreach (string part in parts)
            {
                if (string.IsNullOrEmpty(part.Trim()))
                {
                    continue;
                }

                bool isDelayPunctuation = Regex.IsMatch(part, "^" + delayPunctuationPattern + "$");

                if (isDelayPunctuation)
                {
                    float delay = 0f;
                    switch (part)
                    {
                        case ",":
                        case ";":
                        case ":":
                            delay = commaDelay;
                            break;
                        case ".":
                            delay = periodDelay;
                            break;
                        case "?":
                        case "!":
                            delay = questionExclamationDelay;
                            break;
                    }
                    if (delay > 0)
                    {
                        yield return new WaitForSeconds(delay);
                    }
                }
                else
                {
                    string cleanedChunk = Regex.Replace(part, nonDelayPunctuationPattern, " ");
                    cleanedChunk = cleanedChunk.Trim();

                    if (!string.IsNullOrEmpty(cleanedChunk))
                    {
                        phonemizer.Phonemize(cleanedChunk);

                        yield return new WaitUntil(() => phonemizer.status == ModelStatus.Ready);
                        yield return new WaitUntil(() => piper.status == ModelStatus.Ready);

                        // Wait for all audio samples to be played
                        yield return new WaitUntil(() => !audioSource.isPlaying);
                    }
                }
            }

            status = ModelStatus.Ready;
        }

        private void Awake()
        {
            piper = GetComponentInChildren<PiperModel>();
            phonemizer = GetComponentInChildren<PhonemizerModel>();
            audioSource = GetComponent<AudioSource>();
        }

        private void OnEnable()
        {
            piper.OnStatusChanged += OnModelStatusChanged;
            phonemizer.OnStatusChanged += OnModelStatusChanged;

            phonemizer.OnResponseGenerated += OnPhonemeResponse;
            piper.OnResponseGenerated += OnResponseGenerated;
        }

        private void OnDisable()
        {
            piper.OnStatusChanged -= OnModelStatusChanged;
            phonemizer.OnStatusChanged -= OnModelStatusChanged;

            phonemizer.OnResponseGenerated -= OnPhonemeResponse;
            piper.OnResponseGenerated -= OnResponseGenerated;
        }

        void OnModelStatusChanged(ModelStatus status)
        {
            if (status == ModelStatus.Error)
            {
                StopAllCoroutines();
                this.status = ModelStatus.Error;
            }
        }

        void OnPhonemeResponse(string phonemeString)
        {
            piper.Prompt(phonemeString);
        }

        void OnResponseGenerated(float[] audioChunk, int sampleRate)
        {
            AudioClip clip = AudioClip.Create("GeneratedSpeech", audioChunk.Length, 1, sampleRate, false);
            clip.SetData(audioChunk, 0);

            audioSource.PlayOneShot(clip);
        }
    }
}
