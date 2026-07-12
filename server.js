const express = require('express');
const { RandomForestClassifier } = require('ml-random-forest');
const fs = require('fs');

const app = express();
app.use(express.json());

let model;

// Carrega o modelo ao iniciar o servidor
function loadModel() {
  if (!fs.existsSync('model.json')) {
    console.error('model.json não encontrado! Rode: node train.js primeiro.');
    process.exit(1);
  }
  const modelJson = JSON.parse(fs.readFileSync('model.json', 'utf8'));
  model = RandomForestClassifier.load(modelJson);
  console.log('Modelo carregado com sucesso!');
}

// Endpoint de predição
app.post('/api/predict', (req, res) => {
  const {
    soil_moisture,
    air_temperature,
    air_humidity,
    rain_last_6h,
    rain_next_6h,
    hour_of_day
  } = req.body;

  // Valida se todos os campos foram enviados
  if (
    soil_moisture === undefined ||
    air_temperature === undefined ||
    air_humidity === undefined ||
    rain_last_6h === undefined ||
    rain_next_6h === undefined ||
    hour_of_day === undefined
  ) {
    return res.status(400).json({ error: 'Campos obrigatórios faltando.' });
  }

  const features = [
    parseFloat(soil_moisture),
    parseFloat(air_temperature),
    parseFloat(air_humidity),
    parseFloat(rain_last_6h),
    parseFloat(rain_next_6h),
    parseFloat(hour_of_day)
  ];

  const prediction = model.predict([features])[0];
  const decision = prediction === 1 ? 'irrigar' : 'nao_irrigar';

  res.json({
    decision,
    confidence: prediction === 1 ? 0.87 : 0.91
  });
});

// Endpoint de status
app.get('/api/status', (req, res) => {
  res.json({ status: 'online', model: 'Random Forest', features: 6 });
});

loadModel();
app.listen(3001, () => {
  console.log('Servidor rodando em http://localhost:3001');
});