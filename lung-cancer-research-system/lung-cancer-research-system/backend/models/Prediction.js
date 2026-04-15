const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  datasetId: { type: mongoose.Schema.Types.ObjectId, ref: 'Dataset' },
  modelUsed: { type: String, required: true },
  balancingMethod: { type: String, default: 'CTGAN' },
  prediction: { type: String },
  accuracy: { type: Number },
  precision: { type: Number },
  recall: { type: Number },
  f1Score: { type: Number },
  inputFeatures: { type: Object },
  allResults: { type: Object },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Prediction', PredictionSchema);
