const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function loadData() {
  const numSample = 1000;
  const xTrain = tf.randomNormal([numSample, 28, 28, 1]);
  const yTrain = tf.oneHot(tf.floor(tf.randomUniform([numSample], 0, 10)), 10); // Changed 100 → 10
  return { xTrain, yTrain };
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3, 
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.conv2d({ 
    filters: 64,              
    kernelSize: 3,
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy', // Fixed typo: "categorical Crossentropy"
    metrics: ['accuracy']
  });

  return model;
}

async function train() {
  const { xTrain, yTrain } = await loadData();
  const model = createModel(); // Fixed typo: "creatModel" → "createModel"
  await model.fit(xTrain, yTrain, {
    epochs: 10,
    batchSize: 16,
    validationSplit: 0.1,
    callbacks: tf.node.tensorBoard('./logs')
  });
  await model.save('file://./my-model');
}

train();

