#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
NAMESPACE="ephapsys-staging"
MONGO_POD="mongo-744bbdc5fd-krs5n"
DB_NAME="ephapsys"

echo "🔍 Inspecting remote MongoDB in namespace=$NAMESPACE, pod=$MONGO_POD, db=$DB_NAME"
echo "------------------------------------------------------------"

# sanity check pod
kubectl get pod "$MONGO_POD" -n "$NAMESPACE" >/dev/null 2>&1 || { echo "❌ Mongo pod not found"; exit 1; }

kubectl exec -n "$NAMESPACE" "$MONGO_POD" -- mongosh --quiet --eval "
const db = db.getSiblingDB('$DB_NAME');

print('📚 Collections:');
printjson(db.getCollectionNames());

const metricCollections = db.getCollectionNames().filter(n => /metric|telemetry|event/i.test(n));
print('\\n🔎 Metric-like collections:', metricCollections);

function printJSON(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return obj.toString(); }
}

function printModel(m) {
  const name = m.name || m._id;
  const mod = m.Modulation || {};
  const desired = m.DesiredModulation || {};
  const metricsRoot = m.metrics || {};
  const baseline = metricsRoot.baseline || {};
  const modulated = metricsRoot.modulated || {};
  const stream = mod.metrics_stream || [];
  const lenStream = stream.length;

  print('\\n📄 Model: ' + name);
  print('   • Status: ' + (m.status || '—'));
  print('   • Kind: ' + (m.kind || '—'));
  print('   • Job ID: ' + (mod.job_id || '—'));
  print('   • Modulation.status: ' + (mod.status || '—'));
  print('   • Desired.variant: ' + (desired.variant || '—'));
  const maxSteps = desired.kpi?.maxSteps || mod.kpi?.maxSteps || '—';
  print('   • maxSteps: ' + maxSteps);
  print('   • metrics_stream length: ' + lenStream);
  if (lenStream > 0) {
    const maxStep = Math.max(...stream.map(e => e?.step || 0));
    print('   • metrics_stream max step: ' + maxStep);
    print('   • stream head: ' + printJSON(stream.slice(0, 2)));
    print('   • stream tail: ' + printJSON(stream.slice(-2)));
  } else {
    print('   • (no metrics_stream entries)');
  }

  if (Object.keys(baseline).length) {
    const summary = Object.fromEntries(
      Object.entries(baseline).map(([k, v]) => [k, Array.isArray(v) ? v.length : 0])
    );
    print('   • baseline KPIs (count): ' + printJSON(summary));
  } else {
    print('   • baseline KPIs: (none)');
  }

  if (Object.keys(modulated).length) {
    print('   • modulated snapshot keys: ' + Object.keys(modulated));
  } else {
    print('   • modulated snapshot: (none)');
  }
}

const cur = db.models.find({}, {
  name:1, kind:1, status:1, metrics:1,
  DesiredModulation:1, Modulation:1
}).limit(10);
while (cur.hasNext()) { printModel(cur.next()); }

print('\\n✅ Read-only audit finished.');
"
