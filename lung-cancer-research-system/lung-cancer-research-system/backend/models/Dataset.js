const mongoose = require('mongoose');

const DatasetSchema = new mongoose.Schema({
  datasetName: { type: String, required: true },
  filename: { type: String, required: true },
  uploadedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  uploadDate: { type: Date, default: Date.now },
  rows: { type: Number, default: 0 },
  columns: { type: Number, default: 0 },
  features: [String],
  status: { type: String, enum: ['uploaded', 'preprocessed', 'augmented'], default: 'uploaded' },
  stats: { type: Object, default: {} }
});

module.exports = mongoose.model('Dataset', DatasetSchema);
