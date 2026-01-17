const { runBacktest } = require('./backtest-runner');
const fs = require('fs');
const tf = require('@tensorflow/tfjs');

// 1. Define the Search Space
const epochsOptions = [20, 50]; // Keep small for demo, increase on GPU
const lookbackOptions = [30, 60];
const thresholdOptions = [0.003, 0.005, 0.008]; // 0.3%, 0.5%, 0.8%

const featureSets = [
    {
        name: 'Price Only',
        features: ['close', 'volume', 'rsi']
    },
    {
        name: 'Full Macro',
        features: ['close', 'rsi', 'macd', 'sma', 'volume', 'nifty', 'oil', 'gold', 'silver', 'copper', 'us10y', 'dxy']
    }
];

async function runOptimizer() {
    // Init Backend
    try {
        // Try to load the Node.js backend (C++ / GPU)
        // This works for both @tensorflow/tfjs-node and @tensorflow/tfjs-node-gpu
        const tfNode = require('@tensorflow/tfjs-node-gpu'); 
        console.log(`✅ Loaded TensorFlow Node GPU Backend: ${tf.version.tfjs}`);
    } catch(e) { 
        try {
             require('@tensorflow/tfjs-node'); // Fallback to CPU C++
             console.log('⚠️ Loaded TensorFlow Node CPU Backend (No GPU found)');
        } catch (e2) {
             console.log('⚠️ C++ bindings not found. Using default JS (Slow).'); 
        }
    }

    console.log('🚀 ZEUS OPTIMIZER: Starting Grid Search...');
    const results = [];

    // Grid Loop
    for (const epochs of epochsOptions) {
        for (const lookback of lookbackOptions) {
            for (const thresh of thresholdOptions) {
                for (const fSet of featureSets) {
                    
                    console.log(`\n🧪 Testing: ${fSet.name} | Epochs: ${epochs} | Lookback: ${lookback} | Thresh: ${(thresh*100).toFixed(1)}%`);
                    
                    try {
                        const res = await runBacktest({
                            epochs: epochs,
                            lookBackPeriod: lookback,
                            threshold: thresh,
                            features: fSet.features
                        });

                        console.log(`   👉 Result: ROI ${res.roi}% | Trades: ${res.trades} | WR: ${res.winRate}%`);
                        
                        results.push({
                            config: {
                                name: fSet.name,
                                epochs,
                                lookback,
                                threshold: thresh
                            },
                            metrics: {
                                roi: res.roi,
                                trades: res.trades,
                                winRate: res.winRate
                            }
                        });

                    } catch (err) {
                        console.error('   ❌ Failed:', err.message);
                    }
                }
            }
        }
    }

    // Sort by ROI
    results.sort((a, b) => b.metrics.roi - a.metrics.roi);

    console.log('\n=======================================\n');
    console.log('🏆 BEST CONFIGURATIONS:');
    results.slice(0, 5).forEach((r, i) => {
        console.log(`#${i+1}: ROI ${r.metrics.roi}% | WR ${r.metrics.winRate}% | ${r.config.name} (E:${r.config.epochs}, L:${r.config.lookback}, T:${r.config.threshold})`);
    });

    // Save to file
    fs.writeFileSync('optimization-results.json', JSON.stringify(results, null, 2));
    console.log('\n✅ Results saved to optimization-results.json');
}

runOptimizer();
