kubectl exec -n ephapsys-staging mongo-744bbdc5fd-krs5n -- mongosh --quiet --eval '
db = db.getSiblingDB("ephapsys");
const modelId = ObjectId("68efcedeb1bd4d7eebe82758"); // GPT-2 model ID
const count = db.telemetry.countDocuments({ model_id: modelId });
print(`Telemetry entries for model ${modelId}:`, count);
if (count > 0) {
  const sample = db.telemetry.find({ model_id: modelId }, { step:1, accuracy:1, loss:1, perplexity:1 })
    .sort({ step: 1 })
    .limit(5)
    .toArray();
  const last = db.telemetry.find({ model_id: modelId }, { step:1, accuracy:1, loss:1, perplexity:1 })
    .sort({ step: -1 })
    .limit(2)
    .toArray();
  print("First few steps:");
  printjson(sample);
  print("Last few steps:");
  printjson(last);
} else {
  print("⚠️  No telemetry found for this model — metrics_stream may not have been written.");
}
'
