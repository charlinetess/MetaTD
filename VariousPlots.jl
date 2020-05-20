
# plot means latencies
using PyPlot
# create mean latencies per rats : 
Mean_Lat=[ [mean([currentexperiment[indexrat][indexday][indextrial].latency for indexday=1:numberofdaystest] ) for indextrial=1:numberoftrialstest ] for indexrat=1:numberofrats ];
# Calculate the error bar : 
uppererror = [std([Mean_Lat[indexrat][indextrial] for indexrat in 1:numberofrats]; corrected=false)./sqrt(numberofrats) for indextrial in 1:numberoftrialstest] ;
lowererror = [std([Mean_Lat[indexrat][indextrial] for indexrat in 1:numberofrats]; corrected=false)./sqrt(numberofrats) for indextrial in 1:numberoftrialstest] ;
errs=[lowererror,uppererror]; # gather 


clf()
ioff()
fig = figure("Test plot latencies",figsize=(9,9))

ax=gca() 


#for indextrial=1:numberoftrialstest
PyPlot.plot((1:1:numberoftrialstest), [mean([Mean_Lat[indexrat][indextrial] for indexrat in 1:numberofrats]) for indextrial in 1:numberoftrialstest], marker="None",linestyle="-",color="darkgreen",label="Base Plot")
PyPlot.errorbar((1:1:numberoftrialstest),[mean([Mean_Lat[indexrat][indextrial] for indexrat in 1:numberofrats]) for indextrial in 1:numberoftrialstest],yerr=errs,fmt="o",color="k")

mx = matplotlib.ticker.MultipleLocator(1) # Define interval of minor ticks
ax.xaxis.set_major_locator(mx) # Set interval of minor ticks

ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["bottom"].set_visible("False")
ax.spines["left"].set_visible("False")
  
ax[:set_ylim]([0,ymax])
xmin, xmax = ax.get_xlim() 
ymin, ymax = ax.get_ylim()
# get width and height of axes object to compute 
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height
# manual arrowhead width and length
hw = 1/20*(ymax-ymin) 
hl = 1/20*(xmax-xmin)
lw = 1 # axis line width
ohg = 0.3 # arrow overhang
# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
ax.arrow(xmin, ymin, xmax-xmin, 0.,length_includes_head= "True", fc="k", ec="k", lw = lw,head_width=hw, head_length=hl, overhang = ohg,  clip_on = "False") 

ax.arrow(xmin, ymin, 0., ymax-ymin,length_includes_head= "True", fc="k", ec="k", lw = lw, head_width=yhw, head_length=yhl, overhang = ohg,  clip_on = "False")
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))



#rc("font", family="serif",size=16)
#title("One-shot learning in artificial watermaze")
xlabel("Trials ")#, fontsize=18);
ylabel("Mean latencies")#, fontsize=18)

show()




# plot all latencies :
using PyPlot

for indexday=1:numberofdaystest
# Calculate the lower value for the error bar : 
uppererror = [std([currentexperiment[indexrat][indexday][indextrial].latency for indexrat in 1:numberofrats]; corrected=false)./sqrt(numberofrats) for indextrial in 1:numberoftrialstest] ;
lowererror = [std([currentexperiment[indexrat][indexday][indextrial].latency for indexrat in 1:numberofrats]; corrected=false)./sqrt(numberofrats) for indextrial in 1:numberoftrialstest] ;

errs=[lowererror,uppererror];

PyPlot.plot((indexday-1)*numberoftrialstest.+(1:numberoftrialstest), [mean([currentexperiment[indexrat][indexday][indextrial].latency for indexrat in 1:numberofrats]) for indextrial in 1:numberoftrialstest], marker="None",linestyle="-",color="darkgreen",label="Base Plot")
PyPlot.errorbar((indexday-1)*numberoftrialstest.+(1:numberoftrialstest),[mean([currentexperiment[indexrat][indexday][indextrial].latency for indexrat in 1:numberofrats]) for indextrial in 1:numberoftrialstest],yerr=errs,fmt="o",color="k")

rc("font", family="serif",size=16)
title("One-shot learning in artificial watermaze")
xlabel("Trials ", fontsize=18);
ylabel("Time to the goal (s)", fontsize=18)
end
show()






# plot every platform position 
theta=0:pi/50:(2*pi+pi/50); # to plot circles 

for indexstrategy=1:numberofstrategies
	plot(parameters[:Xplatform][indexstrategy]).+parameters[:r]*cos.(theta),parameters[:Yplatform][indexstrategy]).+parameters[:r]*sin.(theta),color="darkred");

end
	plot(parameters[:R]*cos.(theta),parameters[:R]*sin.(theta),"darkgreen",lw=2)

show()

