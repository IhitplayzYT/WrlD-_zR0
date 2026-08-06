import helper
import data.augment as augment
import data.generate as generate
import ingestors.csv as CsvIngest
import ingestors.pdf as PdfIngest
import ingestors.text as TextIngest

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    if clargs.dbg:
        print(clargs)
    # Generates a random sampled data for stock based on count of stock_data files provided
    # The data is randomly scaled shuffled sampled and trimmed to clargs.each_day_movement * clargs.episode_day_len with a total of clargs.mx_episodes dataframes 
    stock_map = {}
    cur = 0
    if clargs.stock_data:
        for k,x in clargs.stock_data.items():
            if x.endswith(".csv") or x.endswith(".xls") or x.endswith(".xlsx"):
                stock_map[k] = augment.augment_stock_data(CsvIngest.ingest_excel(x),clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
            elif x.endswith(".pdf"):
                stock_map[k] = augment.augment_stock_data(PdfIngest.ingest_pdf(x),clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
            else:
                stock_map[k] = augment.augment_stock_data(TextIngest.ingest_txt(x),clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
    else:
        stock_map[f"SYNTH{cur}"] = augment.augment_stock_data(generate.generate_stock()["prices"])
        cur += 1
         
     