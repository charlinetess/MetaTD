
using LaTeXStrings

using LinearAlgebra
using Statistics
using JLD2 
using FileIO
using PyPlot
using PyCall
@pyimport matplotlib.animation as anim
# anim=pyimport("matplotlib.animation") # I thnk new way of doing it 
using IJulia
zoom= pyimport("mpl_toolkits.axes_grid1.inset_locator") # for embeddded plots 
@pyimport matplotlib.projections as proj




rats=load("Meta_newParameter.jld2"); 
parameters=rats["parameters"];
meta_parameters=rats["meta_parameters"]
features=rats["features"];
data=rats["data"]; # Fields are  trajectory historySPE latency historyconfidence real_TDerrors estimated_TDerrors historytemperature real_platform indexstrategy 
indexrat=8;
indexday=1;
angles=[2*pi*l/parameters[:NA] for l=1:parameters[:NA]];



# 	
# 	
# 	`7MMM.     ,MMF'                                 `7MMF'            db      mm
# 	  MMMb    dPMM                                     MM             ;MM:     MM
# 	  M YM   ,M MM  .gP"Ya   ,6"Yb.  `7MMpMMMb.        MM            ,V^MM.  mmMMmm .gP"Ya `7MMpMMMb.  ,p6"bo `7M'   `MF'
# 	  M  Mb  M' MM ,M'   Yb 8)   MM    MM    MM        MM           ,M  `MM    MM  ,M'   Yb  MM    MM 6M'  OO   VA   ,V
# 	  M  YM.P'  MM 8M""""""  ,pm9MM    MM    MM        MM      ,    AbmmmqMA   MM  8M""""""  MM    MM 8M         VA ,V
# 	  M  `YM'   MM YM.    , 8M   MM    MM    MM        MM     ,M   A'     VML  MM  YM.    ,  MM    MM YM.    ,    VVV
# 	.JML. `'  .JMML.`Mbmmd' `Moo9^Yo..JMML  JMML.    .JMMmmmmMMM .AMA.   .AMMA.`Mbmo`Mbmmd'.JMML  JMML.YMbmd'     ,V
# 	                                                                                                             ,V
# 	                                                                                                          OOb"

# create mean latencies per rats : 
Mean_Lat=[ [mean([data[indexrat][indexday][indextrial].latency for indexday=1:meta_parameters[:numberofdaystest] ]) for indextrial=1:meta_parameters[:numberoftrialstest ]] for indexrat=1:features[:numberofrats] ];
# Calculate the error bar : 
uppererror = [std([Mean_Lat[indexrat][indextrial] for indexrat in 1:features[:numberofrats]]; corrected=false)./sqrt(features[:numberofrats]) for indextrial in 1:meta_parameters[:numberoftrialstest ]] ;
lowererror = [std([Mean_Lat[indexrat][indextrial] for indexrat in 1:features[:numberofrats]]; corrected=false)./sqrt(features[:numberofrats]) for indextrial in 1:meta_parameters[:numberoftrialstest ]] ;
errs=[lowererror,uppererror]; # gather 


clf()
ioff()
fig = figure("Test plot latencies",figsize=(9,9))

ax=gca() 


#for indextrial=1:numberoftrialstest
PyPlot.plot((1:1:meta_parameters[:numberoftrialstest ]), [mean([Mean_Lat[indexrat][indextrial] for indexrat in 1:features[:numberofrats]]) for indextrial in 1:meta_parameters[:numberoftrialstest ]], marker="None",linestyle="-",color="darkgreen",label="Base Plot")
PyPlot.errorbar((1:1:meta_parameters[:numberoftrialstest ]),[mean([Mean_Lat[indexrat][indextrial] for indexrat in 1:features[:numberofrats]]) for indextrial in 1:meta_parameters[:numberoftrialstest ]],yerr=errs,fmt="o",color="k")

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




# 	
# 	                ,,    ,,                                                            ,,
# 	      db      `7MM  `7MM      `7MMF'               mm                               db
# 	     ;MM:       MM    MM        MM                 MM
# 	    ,V^MM.      MM    MM        MM         ,6"Yb.mmMMmm .gP"Ya `7MMpMMMb.  ,p6"bo `7MM  .gP"Ya  ,pP"Ybd
# 	   ,M  `MM      MM    MM        MM        8)   MM  MM  ,M'   Yb  MM    MM 6M'  OO   MM ,M'   Yb 8I   `"
# 	   AbmmmqMA     MM    MM        MM      ,  ,pm9MM  MM  8M""""""  MM    MM 8M        MM 8M"""""" `YMMMa.
# 	  A'     VML    MM    MM        MM     ,M 8M   MM  MM  YM.    ,  MM    MM YM.    ,  MM YM.    , L.   I8
# 	.AMA.   .AMMA..JMML..JMML.    .JMMmmmmMMM `Moo9^Yo.`Mbmo`Mbmmd'.JMML  JMML.YMbmd' .JMML.`Mbmmd' M9mmmP'
# 	
# 	


clf()
ioff()
let fig,ax

	fig = figure("Test plot latencies",figsize=(9,9))

	ax = fig[:add_subplot](1,1,1)


	global labels_code=[] # init labels 
	for indexday=1:meta_parameters[:numberofdaystest ]

	# Calculate the lower value for the error bar : 
	uppererror = [std([data[indexrat][indexday][indextrial].latency for indexrat in 1:features[:numberofrats]]; corrected=false)./sqrt(features[:numberofrats]) for indextrial in 1:meta_parameters[:numberoftrialstest]] ;
	lowererror = [std([data[indexrat][indexday][indextrial].latency for indexrat in 1:features[:numberofrats]]; corrected=false)./sqrt(features[:numberofrats]) for indextrial in 1:meta_parameters[:numberoftrialstest]] ;

	errs=[lowererror,uppererror];

	PyPlot.plot((indexday-1)*meta_parameters[:numberoftrialstest].+(1:meta_parameters[:numberoftrialstest]), [mean([data[indexrat][indexday][indextrial].latency for indexrat in 1:features[:numberofrats]]) for indextrial in 1:meta_parameters[:numberoftrialstest]], marker="None",linestyle="-",color="darkgreen",label="Base Plot")
	PyPlot.errorbar((indexday-1)*meta_parameters[:numberoftrialstest].+(1:meta_parameters[:numberoftrialstest]),[mean([data[indexrat][indexday][indextrial].latency for indexrat in 1:features[:numberofrats]]) for indextrial in 1:meta_parameters[:numberoftrialstest]],yerr=errs,fmt="o",color="k")


	global labels_code=vcat(labels_code,collect(1:1:meta_parameters[:numberoftrialstest]))
	println(labels_code)
	end
	 


	mx = matplotlib.ticker.MultipleLocator(1) # Define interval of minor ticks
	ax.xaxis.set_major_locator(mx) # Set interval of minor ticks

	ax.spines["top"].set_color("none")
	ax.spines["right"].set_color("none")
	ax.spines["bottom"].set_visible("False")
	ax.spines["left"].set_visible("False")
	  ymin, ymax = ax.get_ylim()
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
	ylabel("Latencies")#, fontsize=18)

	#labels = [item.get_text() for item in ax.get_xticklabels()]
	#labels[1] = labels_code
	labels_code=vcat("","",labels_code)
	ax.set_xticklabels(labels_code)

	SMALL_SIZE = 10
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 30

	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

	show()
end # end let fig, ax


# plot every platform position 
theta=0:pi/50:(2*pi+pi/50); # to plot circles 

for indexstrategy=1:meta_parameters[:numberofstrategies]
	plot(parameters[:Xplatform][indexstrategy]).+parameters[:r]*cos.(theta),parameters[:Yplatform][indexstrategy]).+parameters[:r]*sin.(theta),color="darkred");

end
	plot(parameters[:R]*cos.(theta),parameters[:R]*sin.(theta),"darkgreen",lw=2)

show()



# 	
# 	
# 	 .M"""bgd `7MM"""Mq.`7MM"""YMM
# 	,MI    "Y   MM   `MM. MM    `7
# 	`MMb.       MM   ,M9  MM   d
# 	  `YMMNq.   MMmmdM9   MMmmMM
# 	.     `MM   MM        MM   Y  ,
# 	Mb     dM   MM        MM     ,M
# 	P"Ybmmd"  .JMML.    .JMMmmmmMMM
# 	
# 	

indexrat=1;
indexday=3;
indextrial=1;
angles=[2*pi*l/parameters[:NA] for l=1:parameters[:NA]];


fig = plt.figure("MyFigure")

ax2=gca()

plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),[data[indexrat][indexday][indextrial].historySPE[j][data[indexrat][indexday][indextrial].indexstrategy] for j=1:size(data[indexrat][indexday][indextrial].trajectory,1)],"seagreen")

# plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),data[indexrat][indexday][indextrial].historySPE,"seagreen")
ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
# ax2.set_xticklabels(labels) # labels of the ticks 
majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
#setp(ax2.get_yticklabels(),visible=false)
ax2.spines["top"].set_color("none")
ax2.spines["right"].set_color("none")
ax2.spines["bottom"].set_visible("False")
ax2.spines["left"].set_visible("False")
show()



# 	
# 	
# 	MMP""MM""YMM                                                         mm
# 	P'   MM   `7                                                         MM
# 	     MM  .gP"Ya `7MMpMMMb.pMMMb. `7MMpdMAo.  .gP"Ya `7Mb,od8 ,6"Yb.mmMMmm `7MM  `7MM  `7Mb,od8 .gP"Ya
# 	     MM ,M'   Yb  MM    MM    MM   MM   `Wb ,M'   Yb  MM' "'8)   MM  MM     MM    MM    MM' "',M'   Yb
# 	     MM 8M""""""  MM    MM    MM   MM    M8 8M""""""  MM     ,pm9MM  MM     MM    MM    MM    8M""""""
# 	     MM YM.    ,  MM    MM    MM   MM   ,AP YM.    ,  MM    8M   MM  MM     MM    MM    MM    YM.    ,
# 	   .JMML.`Mbmmd'.JMML  JMML  JMML. MMbmmd'   `Mbmmd'.JMML.  `Moo9^Yo.`Mbmo  `Mbod"YML..JMML.   `Mbmmd'
# 	                                   MM
# 	                                 .JMML.


indexrat=1;
indexday=3;
indextrial=1;
angles=[2*pi*l/parameters[:NA] for l=1:parameters[:NA]];


fig = plt.figure("MyFigure")

ax2=gca()

plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),data[indexrat][indexday][indextrial].historytemperature,"seagreen")
ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
# ax2.set_xticklabels(labels) # labels of the ticks 
majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
#setp(ax2.get_yticklabels(),visible=false)
ax2.spines["top"].set_color("none")
ax2.spines["right"].set_color("none")
ax2.spines["bottom"].set_visible("False")
ax2.spines["left"].set_visible("False")

show()



# 	
# 	                                   ,...,,        ,,
# 	  .g8"""bgd                      .d' ""db      `7MM
# 	.dP'     `M                      dM`             MM
# 	dM'       ` ,pW"Wq.`7MMpMMMb.   mMMmm`7MM   ,M""bMM  .gP"Ya `7MMpMMMb.  ,p6"bo   .gP"Ya
# 	MM         6W'   `Wb MM    MM    MM    MM ,AP    MM ,M'   Yb  MM    MM 6M'  OO  ,M'   Yb
# 	MM.        8M     M8 MM    MM    MM    MM 8MI    MM 8M""""""  MM    MM 8M       8M""""""
# 	`Mb.     ,'YA.   ,A9 MM    MM    MM    MM `Mb    MM YM.    ,  MM    MM YM.    , YM.    ,
# 	  `"bmmmd'  `Ybmd9'.JMML  JMML..JMML..JMML.`Wbmd"MML.`Mbmmd'.JMML  JMML.YMbmd'   `Mbmmd'
# 	
# 	


indexrat=8;
indexday=1;
indextrial=1;

angles=[2*pi*l/parameters[:NA] for l=1:parameters[:NA]];


fig = plt.figure("MyFigure")

ax2=gca()

plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),data[indexrat][indexday][indextrial].historyconfidence,"seagreen")
ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
# ax2.set_xticklabels(labels) # labels of the ticks 
majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
#setp(ax2.get_yticklabels(),visible=false)
ax2.spines["top"].set_color("none")
ax2.spines["right"].set_color("none")
ax2.spines["bottom"].set_visible("False")
ax2.spines["left"].set_visible("False")
show()


# 	
# 	                          ,,
# 	MMP""MM""YMM              db                    mm
# 	P'   MM   `7                                    MM
# 	     MM  `7Mb,od8 ,6"Yb.`7MM  .gP"Ya   ,p6"bo mmMMmm ,pW"Wq.`7Mb,od8 `7M'   `MF'
# 	     MM    MM' "'8)   MM  MM ,M'   Yb 6M'  OO   MM  6W'   `Wb MM' "'   VA   ,V
# 	     MM    MM     ,pm9MM  MM 8M"""""" 8M        MM  8M     M8 MM        VA ,V
# 	     MM    MM    8M   MM  MM YM.    , YM.    ,  MM  YA.   ,A9 MM         VVV
# 	   .JMML..JMML.  `Moo9^Yo.MM  `Mbmmd'  YMbmd'   `Mbmo`Ybmd9'.JMML.       ,V
# 	                       QO MP                                            ,V
# 	                       `bmP                                          OOb"


indexrat=9;
indexday=1;
indextrial=1;


argument=0:pi/50:2pi+pi/50;
xplat=parameters[:r]*cos.(argument);
yplat=parameters[:r]*sin.(argument);
xmaze=parameters[:R]*cos.(argument);
ymaze=parameters[:R]*sin.(argument);


fig = plt.figure("MyFigure")

ax=gca()
ax[:set_ylim]([-101,101])
ax[:set_xlim]([-101,101])
xlabel("X")
ylabel("Y")
plot(parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat,color="slateblue",lw=2)
plot(data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2] .+ yplat,color="darkgoldenrod",lw=2)
plot(xmaze,ymaze,color="darkgrey",lw=1)
plot(data[indexrat][indexday][indextrial].trajectory[:,1],data[indexrat][indexday][indextrial].trajectory[:,2],color="darkslategray", lw=2)
ax.set_axis_off()

show()







# 	
# 	                                                                                              ,,                                         ,...
# 	 .M"""bgd `7MM"""Mq.`7MM"""YMM         MMP""MM""YMM `7MM"""Mq.                              `7MM        .g8"""bgd                      .d' ""
# 	,MI    "Y   MM   `MM. MM    `7         P'   MM   `7   MM   `MM.                               MM      .dP'     `M                      dM`
# 	`MMb.       MM   ,M9  MM   d                MM        MM   ,M9      ,6"Yb.  `7MMpMMMb.   ,M""bMM      dM'       ` ,pW"Wq.`7MMpMMMb.   mMMmm
# 	  `YMMNq.   MMmmdM9   MMmmMM                MM        MMmmdM9      8)   MM    MM    MM ,AP    MM      MM         6W'   `Wb MM    MM    MM
# 	.     `MM   MM        MM   Y  ,             MM        MM            ,pm9MM    MM    MM 8MI    MM      MM.        8M     M8 MM    MM    MM
# 	Mb     dM   MM        MM     ,M ,,          MM        MM           8M   MM    MM    MM `Mb    MM      `Mb.     ,'YA.   ,A9 MM    MM    MM
# 	P"Ybmmd"  .JMML.    .JMMmmmmMMM dg        .JMML.    .JMML.         `Moo9^Yo..JMML  JMML.`Wbmd"MML.      `"bmmmd'  `Ybmd9'.JMML  JMML..JMML.
# 	                                ,j
# 	                               ,'

indexrat=16;
indexday=3;
indextrial=1;
angles=[2*pi*l/parameters[:NA] for l=1:parameters[:NA]];


clf()

fig = plt.figure("MyFigure")


ax3=subplot(3,1,1)

plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),[data[indexrat][indexday][indextrial].historySPE[j][data[indexrat][indexday][indextrial].indexstrategy] for j=1:size(data[indexrat][indexday][indextrial].trajectory,1)],"seagreen")

# plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),data[indexrat][indexday][indextrial].historySPE,"seagreen")
ax3[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
# ax2.set_xticklabels(labels) # labels of the ticks 
majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
ax3.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
#setp(ax2.get_yticklabels(),visible=false)
ax3.spines["top"].set_color("none")
ax3.spines["right"].set_color("none")
ax3.spines["bottom"].set_visible("False")
ax3.spines["left"].set_visible("False")


ax2=subplot(3,1,2)

plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),data[indexrat][indexday][indextrial].historyconfidence,"seagreen")
ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
# ax2.set_xticklabels(labels) # labels of the ticks 
majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
#setp(ax2.get_yticklabels(),visible=false)
ax2.spines["top"].set_color("none")
ax2.spines["right"].set_color("none")
ax2.spines["bottom"].set_visible("False")
ax2.spines["left"].set_visible("False")



ax=subplot(3,1,3)

plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),data[indexrat][indexday][indextrial].historytemperature,"seagreen")
ax[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
# ax2.set_xticklabels(labels) # labels of the ticks 
majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
#setp(ax2.get_yticklabels(),visible=false)
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["bottom"].set_visible("False")
ax.spines["left"].set_visible("False")

show()

# 	
# 	                               ,,             ,,
# 	               mm              db           `7MM
# 	               MM                             MM
# 	 pd*"*b.     mmMMmm `7Mb,od8 `7MM   ,6"Yb.    MM  ,pP"Ybd
# 	(O)   j8       MM     MM' "'   MM  8)   MM    MM  8I   `"
# 	    ,;j9       MM     MM       MM   ,pm9MM    MM  `YMMMa.
# 	 ,-='          MM     MM       MM  8M   MM    MM  L.   I8
# 	Ammmmmmm       `Mbmo.JMML.   .JMML.`Moo9^Yo..JMML.M9mmmP'
# 	
# 	

fig=figure()
ax2=fig.gca()
#subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

x=0:1:(size(vcat(data[indexrat][indexday][1].historyconfidence[:],data[indexrat][indexday][2].historyconfidence[:]),1)-1)# gather the whole length of trial 1 and trial 2


plot(x,vcat(data[indexrat][indexday][1].historyconfidence[:],data[indexrat][indexday][2].historyconfidence[:]),color="seagreen",lw=4)  # confidence

ax2[:set_xlim]([-meta_parameters[:dt],length(x)+2*meta_parameters[:dt]])

ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][1].historyconfidence,data[indexrat][indexday][2].historyconfidence))-minimum(vcat(data[indexrat][indexday][1].historyconfidence,data[indexrat][indexday][2].historyconfidence))/100,maximum(vcat(data[indexrat][indexday][1].historyconfidence,data[indexrat][indexday][2].historyconfidence))+maximum(vcat(data[indexrat][indexday][1].historyconfidence,data[indexrat][indexday][2].historyconfidence))/10])

ymax=ax2.get_ylim()
# find minimum of SPE: 
x1=findall(x->x==minimum(data[indexrat][indexday][1].historyconfidence[:]),data[indexrat][indexday][1].historyconfidence[:])
# plot([x1,x1],[ymax[1],ymax[1]+1],color="lightseagreen",lw=3) # markers

x2=findall(x->x==maximum(vcat(data[indexrat][indexday][1].historyconfidence,data[indexrat][indexday][2].historyconfidence)),vcat(data[indexrat][indexday][1].historyconfidence,data[indexrat][indexday][2].historyconfidence))
# plot([x2,x2],[ymax[1],ymax[1]+1],color="lightseagreen",lw=3) # markers 

x3=x2.-60; # just a point in the middle 
# plot([x3,x3],[ymax[1],ymax[1]+1],color="lightseagreen",lw=3) # markers 

x4=x[end];
# plot([x4,x4],[ymax[1],ymax[1]+1],color="lightseagreen",lw=3) 


majors21 = [ k*200 for k=0:floor((size(data[indexrat][indexday][1].trajectory,1))/200)]

majors22 = [ k*200 for k=0:floor((size(data[indexrat][indexday][2].trajectory,1))/200)]
# vcat(0:meta_parameters[:dt]:(data[indexrat][indexday][1].latency-meta_parameters[:dt]),0:meta_parameters[:dt]:(data[indexrat][indexday][2].latency-meta_parameters[:dt])) 

# floor((size(data[indexrat][indexday][1].trajectory,1)+size(data[indexrat][indexday][2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][1].trajectory,1) + size(data[indexrat][indexday][2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][1].trajectory,1)+size(data[indexrat][indexday][2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
majors2=vcat(sort(vcat(majors21,x1,x2,x3)),sort(vcat(majors22,(x4.-x2)))) # incorporate new points , witht he real timing of x4



ax2.spines["top"].set_color("none")
ax2.spines["right"].set_color("none")
ax2.spines["bottom"].set_visible("False")
ax2.spines["left"].set_visible("False")
ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 

# ax.tick_params(which=, length=4, color='r')

# labels = [item.get_text() for item in ax2.get_xticklabels()]
# define labels  
# majors2labels=vcat([ k*200 for k=0:floor((size(data[indexrat][indexday][1].trajectory,1))/200)],[ k*200 for k=0:floor((size(data[indexrat][indexday][2].trajectory,1))/200)])
# majors2labels=vcat([ k*200 for k=0:floor((size(data[indexrat][indexday][1].trajectory,1))/200)],[ k*200 for k=0:floor((size(data[indexrat][indexday][2].trajectory,1))/200)])

labels=[string(majors2[k]) for k=1:length(majors2) ]; 
text(x1, ymax[1]-ymax[1]/10, "c",size=20)
text(x2, ymax[1]-ymax[1]/10, "e",size=20)
text(x3, ymax[1]-ymax[1]/10, "d",size=20)
text(x4, ymax[1]-ymax[1]/10, "f",size=20)

# labels[1]="0" # for some reason the first label is -200
labels[findall(x->x==x1[1],majors2)[1]] = ""
labels[findall(x->x==x2[1],majors2)[1]] = ""
labels[findall(x->x==x3[1],majors2)[1]] = ""
labels[findall(x->x==(x4.-x2)[1],majors2)[1]] = ""

# labels[end]

ylabel(latexstring("\$\\sigma\$"),size=18)
xlabel("time since trial onset (ms)",size=18)

# # labels[1]="" # no space for both 
# labels[end-1]=""
# labels[9]=""
ax2.set_xticklabels(labels)

show()

# confidence + vertical bars 
# majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
# ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
# ax2.spines["top"].set_color("none")
# ax2.spines["right"].set_color("none")
# ax2.spines["bottom"].set_visible("False")
# ax2.spines["left"].set_visible("False")
