const csv = require('csvtojson');
const { RandomForestClassifier } = require('ml-random-forest');
const fs = require('fs');

async function train() {
  console.log('Carregando dataset...');
  const data = await csv().fromFile('dataset.csv');

  // Separa features (X) e rótulos (y)
  const X = data.map(row => [
    parseFloat(row.soil_moisture),
    parseFloat(row.air_temperature),
    parseFloat(row.air_humidity),
    parseFloat(row.rain_last_6h),
    parseFloat(row.rain_next_6h),
    parseFloat(row.hour_of_day)
  ]);

  const y = data.map(row => row.label === 'irrigar' ? 1 : 0);

  console.log(`Dataset carregado: ${X.length} amostras`);

  // Treina o modelo
  console.log('Treinando Random Forest...');
  const model = new RandomForestClassifier({
    nEstimators: 100,  // 100 árvores
    maxDepth: 10,
    seed: 42
  });

  model.train(X, y);

  // Salva o modelo em JSON
  const modelJson = JSON.stringify(model.toJSON());
  fs.writeFileSync('model.json', modelJson);

  console.log('Modelo treinado e salvo em model.json!');
}

train().catch(console.error);