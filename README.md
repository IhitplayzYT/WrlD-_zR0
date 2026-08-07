# Idea
```
A RLAIF based individuals try to generate maximum profits,with fixed corpus
Those who reach zero/below a threshold are punished and now oppose rest of the models by trying to invest in a way to kill good stock momentum(i.e invests so that others make wrong predictions and go to loss)
Random events can occur after random amount of time, good and bad, those who perform best in good events they will be broadcasted on a newspaper so other profit models try to follow and the malicious ones use it to learn and devise a way to crash the market
Some events can be planned such as days of stock buyback ,wars, facility and infra development,etc
At a periodic time samples the stock graph will be hidden and so models dont have any idea of what current stock prices would be from present to present+k days they will be highly awarded if they invest and generate profits, if losses are there a small penalty to be incurred, if someone doesnt participate they will get large penalty
During the periodic time frame each model will need to generate the stock price line movemement for the k black box days
Line closest to the actual stock price will be awarded as king with a multiplier for rewards gained, foes that predict the market will get Villian Points that can be spent to trigger more bad events at their chosen timings to catch behaviorally fixed pattern investing models offguard
Documents like financial statements etc will be provided to all at regular intervals that may signal to future events and market structure, so NLP and CV will be used by the ingestor to send images and text data and cv for any other interprettion they need to make from images like graphs and trends 
If the economy collapses this terminates one episode if economy continues and shows a growth above x value or y % market cap this marks the end of another episode,on desirable market good guys will be awarded bonus points and if market collapses then bad guys will be rewarded,at a point only max 'k' no of bad guys can exist, with it being a non transitionable resting state(i,e good -> bad but there is no bad -> good transition)
If economy is at a stand still this will cause all the models to be penalized
Latter on the best performing RLAIF model will participate in a real market stock, it will be provided a week of learning time to monitor and watch the stock and then allowed to start investing(not actual money but simulated profits if simulated amount invested)
A summary model will be exposed to the user providing commentary of the actions and news brekaing changes that occured in the market, this will also be sent to each model to add a incentive factor and impart conservative mechanism and learning trends
Some models will not be allowed to interact with stock market on a rare random days/burst of days to add more reality since some days individuals donot see the market or invest 
The models can also be in rare occasions killed/blocked from investing for long burst duration as a very rare ocassion to add extra volatility and incertainity 
If time permits a multi stock simulation would be built so if some stock is interdependetn on another or If one stock is doing bad the model can adapt to a different Portfolio
Each day will be modelled to be 1-2 mins long with a max participants of 20 models to take decisions for today
Use order books to simulate real stock price movement(Use a BtreeMap) 
Compare the movement stock price with the data stock price to see effects of model interacting with Stock Market
Use cross episode rewards using streaks,episode results
Use JSON based logging 
Save the Model Params for fast recreation
Build model traits for each investor
Use timestamps[Some investors are allowed to invest in the first m stock movements of the day some the next n and ...]
Use the SP movements and build a Options pricing graph to see how would an option be priced at each instance
Later add the feature of options
```
