const fs=require('fs');
const {getHistoricalRates}=require('dukascopy-node');
const pairs=['eurusd','gbpusd','audusd','nzdusd','usdjpy','usdcad','usdchf','usdnok','usdsek'];
(async()=>{
 for(const p of pairs){
  console.log('download fixflow',p);
  const csv=await getHistoricalRates({instrument:p,dates:{from:new Date('2020-01-01T00:00:00Z'),to:new Date('2026-08-01T00:00:00Z')},timeframe:'m5',format:'csv',batchSize:48,pauseBetweenBatchesMs:5,retryCount:5,pauseBetweenRetriesMs:300,retryOnEmpty:false,failAfterRetryCount:false});
  fs.writeFileSync(`/tmp/${p}_fixflow_m5.csv`,csv);
  console.log('saved',p,csv.length);
 }
})().catch(e=>{console.error(e);process.exit(1)});
