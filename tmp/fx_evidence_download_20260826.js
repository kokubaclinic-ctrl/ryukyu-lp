const fs=require('fs');
const {getHistoricalRates}=require('dukascopy-node');
const pairs=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','audjpy'];
(async()=>{
 for(const instrument of pairs){
  console.log('download',instrument,new Date().toISOString());
  const csv=await getHistoricalRates({instrument,dates:{from:new Date('2014-01-01T00:00:00Z'),to:new Date('2026-08-01T00:00:00Z')},timeframe:'m5',format:'csv',batchSize:32,pauseBetweenBatchesMs:20,retryCount:6,pauseBetweenRetriesMs:500,retryOnEmpty:false,failAfterRetryCount:false});
  fs.writeFileSync(`/tmp/${instrument}_ev_m5.csv`,csv);
  console.log('saved',instrument,csv.length);
 }
})().catch(e=>{console.error(e);process.exit(1)});
