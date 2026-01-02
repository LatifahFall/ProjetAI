// src/services/api.js

export async function predictFromAudioBlob(audioBlob) {
  // TODO: remplacer par appel backend Flask:
  // const form = new FormData()
  // form.append("file", audioBlob, "audio.webm")
  // const res = await fetch("http://localhost:5000/predict", { method: "POST", body: form })
  // return await res.json()

  // Mock (résultats aléatoires stables)
  await new Promise((r) => setTimeout(r, 900));

  const rand = (min, max) => Math.floor(min + Math.random() * (max - min + 1));
  const ocean = {
    openness: rand(40, 90),
    conscientiousness: rand(40, 90),
    extraversion: rand(40, 90),
    agreeableness: rand(40, 90),
    neuroticism: rand(10, 70),
  };

  const score =
    Math.round(
      (ocean.openness +
        ocean.conscientiousness +
        ocean.extraversion +
        ocean.agreeableness +
        (100 - ocean.neuroticism)) /
        5
    );

  return {
    score,
    ocean,
    advice: "Conseil IA (mock) : adaptez le discours selon le trait dominant.",
  };
}
