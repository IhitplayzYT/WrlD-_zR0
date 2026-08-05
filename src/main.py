import helper
import data.augment as augment
import ingestors.csv as CsvIngest
import ingestors.pdf as PdfIngest
import ingestors.text as TextIngest

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    if clargs.dbg:
        print(clargs)
    stocks = []
    if clargs.stock_data:
        for x in clargs.stock_data:
            if x.endswith(".csv") or x.endswith(".xls") or x.endswith(".xlsx"):
                data.append(CsvIngest.ingest_excel(x))
            elif x.endswith(".pdf"):
                data.append(PdfIngest.ingest_pdf(x))
            else:
                data.append(TextIngest.ingest_txt(x))
    cur = 0
    stock_map = {}
    for s in stocks:
        stock_map[cur] = augment.augment_stock_data(x,clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
        cur += 1
    
    


    
        


            
