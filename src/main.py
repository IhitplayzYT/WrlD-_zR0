import helper



if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    if clargs.dbg:
        print(clargs)
    if clargs.stock_data:
        # TODO: Parse Csv [the amount col only and then divide stock into every_day_movemtns * no_of_days chunks for a total n_episode chunks] 
        # ie minimum data expected is every_day_movements * no_of_days
        # if we have way to much data we will compress to the minimum data len(every_day_movements * no_of_days) since to much data means frequent sampling so consecutive data values would be stagnantly similar
        # to generate the chunks for all the episodes we can do maybe a few passes of random adj day swap with random multiplier like algo
        # Full expected Pipeline
        # ie 1 2 3 4 5 6 7 8
        # ie 2 1 5 4 3 6 7 8 // Initail Random swap
        # 1st episode data -> ie 1.8 1.5 5.9 2.2 3.18 10 8 7.4 // Random percentage add/sub(i.e a +- rand% * a) rand should be low 0.1 - 0.3 but on rare occassion should jump to 0.5-0.6 and after jumping, shall slowly reduce back to 0.1-0.3 after some days
        # For 2nd episode data we ise the 1st episode data(i.e for nth epidsode data we repeat the pipleine on the n-1th episode data)
        # This helps build relatively contigous stock data across episodes allow
        pass
