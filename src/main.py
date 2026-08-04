import helper



if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    if clargs.dbg:
        print(clargs)
    if clargs.stock_data:
        # TODO: Parse Csv [the amount col only and then divide stock into every_day_movemtns * no_of_days chunks for a total n_episode chunks] 
        # ie minimum data expected is every_day_movements * no_of_days
        # if we have way to much data we will compress to the minimum data len since to much data means frequent sampling so consecutive data values would be stagnantly similar
        # to generate the chunks we can do maybe a few passes of random adj day swap like algo
        # ie 1 2 3 4 5 6 7 8
        # ie 2 1 5 4 3 6 7 8
        # each neighbour to be swapped on random this can help create synthetically relatively accurate data
        pass
