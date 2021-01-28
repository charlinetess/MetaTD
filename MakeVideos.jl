# this files generates a video with a trajectory and on the side the confidence level, to see the correspondance between the confidence level and the behaviour of the agent 


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


# 	
# 	           ,,
# 	`7MM"""YMM db
# 	  MM    `7
# 	  MM   d `7MM  .P"Ybmmm `7MM  `7MM  `7Mb,od8 .gP"Ya
# 	  MM""MM   MM :MI  I8     MM    MM    MM' "',M'   Yb
# 	  MM   Y   MM  WmmmP"     MM    MM    MM    8M""""""
# 	  MM       MM 8M          MM    MM    MM    YM.    ,
# 	.JMML.   .JMML.YMMMMMb    `Mbod"YML..JMML.   `Mbmmd'
# 	              6'     dP
# 	              Ybmmmd'



indexrat=8;
indexday=1;
indextrial=1;
indexstep=30;
angles=[2*pi*l/parameters[:NA] for l=1:parameters[:NA]];



argument=0:pi/50:2pi+pi/50;
xplat=parameters[:r]*cos.(argument);
yplat=parameters[:r]*sin.(argument);
xmaze=parameters[:R]*cos.(argument);
ymaze=parameters[:R]*sin.(argument);
X=data[indexrat][indexday][indextrial].trajectory[indexstep,1]
Y=data[indexrat][indexday][indextrial].trajectory[indexstep,2]


# fig3 = plt.figure(figsize=(12,16),constrained_layout=true)
fig3 = PyPlot.figure(figsize=(12,16))#,constrained_layout=true)
gs = matplotlib.gridspec.GridSpec(3,4,figure=fig3)#, width_ratios=[3, 1],height_ratios=[1, 1]) 

ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,3),pycall(pybuiltin("slice"), PyObject, 0,2)))) # we have to define it as a slice object bc 0:2 is not recognised 
ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,2))))
# fig3.set_size_inches(12,16)


# plt.rcParams["figure.constrained_layout.use"] = true
# plt.rcParams["figure.figsize"] = 12, 16


# ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 0,2),pycall(pybuiltin("slice"), PyObject, 1,3)))) # we have to define it as a slice object bc 0:2 is not recognised 


# ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,3),pycall(pybuiltin("slice"), PyObject, 0,3)))) # we have to define it as a slice object bc 0:2 is not recognised 
# ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
ax[:set_ylim]([-101,101])
ax[:set_xlim]([-101,101])
xlabel("X")
ylabel("Y")
scatter(X,Y,color="teal", lw=4)
plot(parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat,color="slateblue",lw=2)
plot(data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2] .+ yplat,color="darkgoldenrod",lw=2)

plot(xmaze,ymaze,color="darkgrey",lw=1)
plot(data[indexrat][indexday][indextrial].trajectory[1:indexstep,1],data[indexrat][indexday][indextrial].trajectory[1:indexstep,2],color="darkslategray", lw=2)
ax.set_axis_off()


# ax2=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 0,2),0)))

# ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,3))))

# ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
confidence=vcat(data[indexrat][indexday][indextrial].historyconfidence[1:indexstep],[NaN for i=1:(size(data[indexrat][indexday][indextrial].trajectory,1)- indexstep)]);
plot(collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),confidence,"seagreen",label="Confidence")
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
# 	              ,,        ,,
# 	`7MMF'   `7MF'db      `7MM
# 	  `MA     ,V            MM
# 	   VM:   ,V `7MM   ,M""bMM  .gP"Ya   ,pW"Wq.
# 	    MM.  M'   MM ,AP    MM ,M'   Yb 6W'   `Wb
# 	    `MM A'    MM 8MI    MM 8M"""""" 8M     M8
# 	     :MM;     MM `Mb    MM YM.    , YA.   ,A9
# 	      VF    .JMML.`Wbmd"MML.`Mbmmd'  `Ybmd9'
# 	
# 	


indexrat=8;
indexday=1;
indextrial=1;

global argument=0:pi/50:2pi+pi/50;
global xplat=parameters[:r]*cos.(argument);
global yplat=parameters[:r]*sin.(argument);
global xmaze=parameters[:R]*cos.(argument);
global ymaze=parameters[:R]*sin.(argument);



fig3 = PyPlot.figure(figsize=(12,16))#,constrained_layout=true)
gs = matplotlib.gridspec.GridSpec(3,4,figure=fig3)#, width_ratios=[3, 1],height_ratios=[1, 1]) 



# fig3.set_size_inches(12,16)


global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,3),pycall(pybuiltin("slice"), PyObject, 0,2)))) # we have to define it as a slice object bc 0:2 is not recognised 

# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 


ax[:set_ylim]([-101,101])
ax[:set_xlim]([-101,101])
global line2=ax[:plot]([],[],color="darkgrey",lw=3)[1] # supposed platform 
global line3=ax[:plot]([],[],color="darkslategray", lw=2)[1]  # maze borders 
global line1=ax[:plot]([],[],color="teal","*", lw=10)[1] # current position 
global line4=ax[:plot]([],[],color="slateblue",lw=3)[1] # trajectory 
global line5=ax[:plot]([],[],color="darkgoldenrod",lw=2)[1] # real platofrm 
#plot(parameters[:Xplatform][data[indexrat][indexday][indextrial].real_platform[1]].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].real_platform[2]] .+ yplat,color="darkgoldenrod",lw=2)
SMALL_SIZE =12
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

ax.set_axis_off()

global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,2))))

# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

global line21=ax2[:plot]([],[],color="seagreen",lw=2)[1]
ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
ax2[:set_ylim]([minimum(data[indexrat][indexday][indextrial].historyconfidence),maximum(data[indexrat][indexday][indextrial].historyconfidence)])

# Define the init function, which draws the first frame (empty, in this case)
function init()
    global line1
    global line2
    global line3
	global line4
    global line21
    global line5
	
	global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
	# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

	line1[:set_data](data[indexrat][indexday][indextrial].trajectory[1,1],data[indexrat][indexday][indextrial].trajectory[1,2])
	line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat)
	line3[:set_data](xmaze,ymaze)
	line4[:set_data](data[indexrat][indexday][indextrial].trajectory[1,1],data[indexrat][indexday][indextrial].trajectory[1,2])
	line5[:set_data](data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2] .+ yplat)
	
	global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

	# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
	# confidence=vcat(data[indexrat][indexday][indextrial].historyconfidence[1],[NaN for j=1:(size(data[indexrat][indexday][indextrial].trajectory,1)- 1)]);
	# line21[:set_data](collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),confidence)
	# line21[:set_data](collect(1:1:1),data[indexrat][indexday][indextrial].historyconfidence[1])
	line21[:set_data]([],[])	
	ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
	ax2[:set_ylim]([minimum(data[indexrat][indexday][indextrial].historyconfidence),maximum(data[indexrat][indexday][indextrial].historyconfidence)])
	majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
	ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
	ax2.spines["top"].set_color("none")
	ax2.spines["right"].set_color("none")
	ax2.spines["bottom"].set_visible("False")
	ax2.spines["left"].set_visible("False")
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    return (line1,line2,line3,line4,line21,line5, Union{})  # Union{} is the new word for None
end



function animate(i)
    global line1
    global line2
    global line3
	global line4
    global line21
    global line5

	i=i+1

	global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

	# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

	line1[:set_data](data[indexrat][indexday][indextrial].trajectory[i,1],data[indexrat][indexday][indextrial].trajectory[i,2])
	line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat)
	line3[:set_data](xmaze,ymaze)
	line4[:set_data](data[indexrat][indexday][indextrial].trajectory[1:i,1],data[indexrat][indexday][indextrial].trajectory[1:i,2])
	line5[:set_data](data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2].+ yplat)

	# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
	global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

	x = (0:1:(i-1))
	line21[:set_data](x,data[indexrat][indexday][indextrial].historyconfidence[1:i])
	ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)])
	ax2[:set_ylim]([minimum(data[indexrat][indexday][indextrial].historyconfidence),maximum(data[indexrat][indexday][indextrial].historyconfidence)])
	majors2 = [floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6)*l for l=0:floor(size(data[indexrat][indexday][indextrial].trajectory,1)/floor(size(data[indexrat][indexday][indextrial].trajectory,1)/6))]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
	ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
	ax2.spines["top"].set_color("none")
	ax2.spines["right"].set_color("none")
	ax2.spines["bottom"].set_visible("False")
	ax2.spines["left"].set_visible("False")
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    return (line1,line2,line3,line4,line21,line5,Union{})  # Union{} is the new word for None
end



mywriter = anim.FFMpegWriter()
myanim = anim.FuncAnimation(fig3, animate, init_func=init, frames=size(data[indexrat][indexday][indextrial].trajectory,1), interval=200)
myanim[:save]("Vid$(indexday).mp4",writer=mywriter) #bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])



# 	
# 	              ,,        ,,                                       ,,             ,,
# 	`7MMF'   `7MF'db      `7MM                   MMP""MM""YMM        db           `7MM
# 	  `MA     ,V            MM                   P'   MM   `7                       MM
# 	   VM:   ,V `7MM   ,M""bMM       pd*"*b.          MM  `7Mb,od8 `7MM   ,6"Yb.    MM  ,pP"Ybd
# 	    MM.  M'   MM ,AP    MM      (O)   j8          MM    MM' "'   MM  8)   MM    MM  8I   `"
# 	    `MM A'    MM 8MI    MM          ,;j9          MM    MM       MM   ,pm9MM    MM  `YMMMa.
# 	     :MM;     MM `Mb    MM       ,-='             MM    MM       MM  8M   MM    MM  L.   I8
# 	      VF    .JMML.`Wbmd"MML.    Ammmmmmm        .JMML..JMML.   .JMML.`Moo9^Yo..JMML.M9mmmP'
# 	
# This shows the evolution trial 1 and 2 


indexrat=8;
indexday=1;
indextrial=1;
indextrial2=2;

global argument=0:pi/50:2pi+pi/50;
global xplat=parameters[:r]*cos.(argument);
global yplat=parameters[:r]*sin.(argument);
global xmaze=parameters[:R]*cos.(argument);
global ymaze=parameters[:R]*sin.(argument);


# fig3 = plt.figure(constrained_layout=true)

# plt.rcParams["figure.constrained_layout.use"] = true
# fig3 = plt.figure(figsize=(12,16))#,constrained_layout=true)

fig3 = PyPlot.figure(figsize=(12,16))#,constrained_layout=true)
gs = matplotlib.gridspec.GridSpec(3,4,figure=fig3)#, width_ratios=[3, 1],height_ratios=[1, 1]) 

#subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,3),pycall(pybuiltin("slice"), PyObject, 0,2)))) # we have to define it as a slice object bc 0:2 is not recognised 


ax[:set_ylim]([-101,101])
ax[:set_xlim]([-101,101])
global line2=ax[:plot]([],[],color="darkgrey",lw=1)[1] # goal of current strategy 
global line3=ax[:plot]([],[],color="darkslategray", lw=2)[1] # maze boundaries 
global line1=ax[:plot]([],[],color="teal","*", lw=5)[1] # agent # scatter(X,Y,color="teal", lw=4)
global line4=ax[:plot]([],[],color="slateblue",lw=2)[1] # trajectory 
global line5=ax[:plot]([],[],color="darkgoldenrod",lw=2)[1] # current goal 
#plot(parameters[:Xplatform][data[indexrat][indexday][indextrial].real_platform[1]].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].real_platform[2]] .+ yplat,color="darkgoldenrod",lw=2)

ax.set_axis_off()


global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,2))))
#subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

global line21=ax2[:plot]([],[],color="seagreen",lw=4)[1] # confidence
global line22=ax2[:plot]([],[],color="lightseagreen",lw=3)[1] # markers 

ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title






# Define the init function, which draws the first frame (empty, in this case)
function init()
    global line1
    global line2
    global line3
	global line4
    global line21
    global line5
	global line22

	# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
	# global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

	line1[:set_data](data[indexrat][indexday][indextrial].trajectory[1,1],data[indexrat][indexday][indextrial].trajectory[1,2])
	line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat)
	line3[:set_data](xmaze,ymaze)
	line4[:set_data](data[indexrat][indexday][indextrial].trajectory[1,1],data[indexrat][indexday][indextrial].trajectory[1,2])
	line5[:set_data](data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2] .+ yplat)

	# global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

	# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
	# confidence=vcat(data[indexrat][indexday][indextrial].historyconfidence[1],[NaN for j=1:(size(data[indexrat][indexday][indextrial].trajectory,1)- 1)]);
	# line21[:set_data](collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),confidence)
	# line21[:set_data](collect(1:1:1),data[indexrat][indexday][indextrial].historyconfidence[1])
	line21[:set_data]([],[])
	line22[:set_data]([],[])	
	ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
	ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
	majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
	ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
	ax2.spines["top"].set_color("none")
	ax2.spines["right"].set_color("none")
	ax2.spines["bottom"].set_visible("False")
	ax2.spines["left"].set_visible("False")
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    return (line1,line2,line3,line4,line21,line5,line22, Union{})  # Union{} is the new word for None
end



function animate(i)
    global line1
    global line2
    global line3
	global line4
    global line21
    global line5
	global line22

	i=i+1

	if i <= size(data[indexrat][indexday][indextrial].trajectory,1)
		# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
		# global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

		line1[:set_data](data[indexrat][indexday][indextrial].trajectory[i,1],data[indexrat][indexday][indextrial].trajectory[i,2])
		line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat)
		line3[:set_data](xmaze,ymaze)
		line4[:set_data](data[indexrat][indexday][indextrial].trajectory[1:i,1],data[indexrat][indexday][indextrial].trajectory[1:i,2])
		line5[:set_data](data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2].+ yplat)

		# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
		# global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

		x = (0:1:(i-1))
		line21[:set_data](x,data[indexrat][indexday][indextrial].historyconfidence[1:i])

		ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
		ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
		
		if i==size(data[indexrat][indexday][indextrial].trajectory,1) # mark the end of trial 1
			ylim=ax2.get_ylim()
			line22[:set_data]([x[size(data[indexrat][indexday][indextrial].trajectory,1)],x[size(data[indexrat][indexday][indextrial].trajectory,1)]],[ylim[1],ylim[2]]) 
		else 
			line22[:set_data]([],[]) # adjusting because the atan function ranges between -pi and pi and we want angles between 0 and 2pi
		end 
		
		majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
		ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
		ax2.spines["top"].set_color("none")
		ax2.spines["right"].set_color("none")
		ax2.spines["bottom"].set_visible("False")
		ax2.spines["left"].set_visible("False")
		plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
		plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
		plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
		plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
		plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
		plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
		plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
	else 
		index=i-size(data[indexrat][indexday][indextrial].trajectory,1);
		# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
		# global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 


		line1[:set_data](data[indexrat][indexday][indextrial2].trajectory[index,1],data[indexrat][indexday][indextrial2].trajectory[index,2])
		line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial2].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial2].indexstrategy] .+ yplat)
		line3[:set_data](xmaze,ymaze)
		line4[:set_data](data[indexrat][indexday][indextrial2].trajectory[1:index,1],data[indexrat][indexday][indextrial2].trajectory[1:index,2])
		line5[:set_data](data[indexrat][indexday][indextrial2].real_platform[1].+xplat,data[indexrat][indexday][indextrial2].real_platform[2].+ yplat)

		# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
		# global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

		x = (0:1:(i-1))
		line21[:set_data](x,vcat(data[indexrat][indexday][indextrial].historyconfidence[:],data[indexrat][indexday][indextrial2].historyconfidence[1:index]))
		ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
		ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
		majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
		ylim=ax2.get_ylim()
		line22[:set_data]([x[size(data[indexrat][indexday][indextrial].trajectory,1)],x[size(data[indexrat][indexday][indextrial].trajectory,1)]],[ylim[1],ylim[2]]) 
		ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
		ax2.spines["top"].set_color("none")
		ax2.spines["right"].set_color("none")
		ax2.spines["bottom"].set_visible("False")
		ax2.spines["left"].set_visible("False")
		plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
		plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
		plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
		plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
		plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
		plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
		plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

	end 	
    return (line1,line2,line3,line4,line21,line5,line22,Union{})  # Union{} is the new word for None
end



mywriter = anim.FFMpegWriter()
myanim = anim.FuncAnimation(fig3, animate, init_func=init, frames=(size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)), interval=200)
myanim[:save]("Vid$(indextrial)_$(indextrial2)_$(indexrat).mp4",writer=mywriter) #bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])



# 	
# 	
# 	`7MMF'        .g8""8q.     .g8""8q. `7MM"""Mq.     `7MMF'        .g8""8q.  MMP""MM""YMM       .g8""8q. `7MM"""YMM     `7MM"""Mq.        db   MMP""MM""YMM  .M"""bgd
# 	  MM        .dP'    `YM. .dP'    `YM. MM   `MM.      MM        .dP'    `YM.P'   MM   `7     .dP'    `YM. MM    `7       MM   `MM.      ;MM:  P'   MM   `7 ,MI    "Y
# 	  MM        dM'      `MM dM'      `MM MM   ,M9       MM        dM'      `MM     MM          dM'      `MM MM   d         MM   ,M9      ,V^MM.      MM      `MMb.
# 	  MM        MM        MM MM        MM MMmmdM9        MM        MM        MM     MM          MM        MM MM""MM         MMmmdM9      ,M  `MM      MM        `YMMNq.
# 	  MM      , MM.      ,MP MM.      ,MP MM             MM      , MM.      ,MP     MM          MM.      ,MP MM   Y         MM  YM.      AbmmmqMA     MM      .     `MM
# 	  MM     ,M `Mb.    ,dP' `Mb.    ,dP' MM             MM     ,M `Mb.    ,dP'     MM          `Mb.    ,dP' MM             MM   `Mb.   A'     VML    MM      Mb     dM
# 	.JMMmmmmMMM   `"bmmd"'     `"bmmd"' .JMML.         .JMMmmmmMMM   `"bmmd"'     .JMML.          `"bmmd"' .JMML.         .JMML. .JMM..AMA.   .AMMA..JMML.    P"Ybmmd"
# 	
# 	
# [16,17,18,19]
# [3,4,5,6]



for indexrat in [16,17,18,19]
	for indexday in [3,4,5,6]
		indextrial=1;
		indextrial2=2;

		global argument=0:pi/50:2pi+pi/50;
		global xplat=parameters[:r]*cos.(argument);
		global yplat=parameters[:r]*sin.(argument);
		global xmaze=parameters[:R]*cos.(argument);
		global ymaze=parameters[:R]*sin.(argument);


		# fig3 = plt.figure(constrained_layout=true)

		# plt.rcParams["figure.constrained_layout.use"] = true
		# fig3 = plt.figure(figsize=(12,16))#,constrained_layout=true)

		fig3 = PyPlot.figure(figsize=(12,16))#,constrained_layout=true)
		gs = matplotlib.gridspec.GridSpec(3,4,figure=fig3)#, width_ratios=[3, 1],height_ratios=[1, 1]) 

		#subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
		global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,3),pycall(pybuiltin("slice"), PyObject, 0,2)))) # we have to define it as a slice object bc 0:2 is not recognised 


		ax[:set_ylim]([-101,101])
		ax[:set_xlim]([-101,101])
		global line2=ax[:plot]([],[],color="darkgrey",lw=1)[1]
		global line3=ax[:plot]([],[],color="darkslategray", lw=2)[1]
		global line1=ax[:plot]([],[],color="teal","*", lw=5)[1] # scatter(X,Y,color="teal", lw=4)
		global line4=ax[:plot]([],[],color="slateblue",lw=2)[1]
		global line5=ax[:plot]([],[],color="darkgoldenrod",lw=2)[1]
		#plot(parameters[:Xplatform][data[indexrat][indexday][indextrial].real_platform[1]].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].real_platform[2]] .+ yplat,color="darkgoldenrod",lw=2)

		ax.set_axis_off()


		global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,2))))
		#subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

		global line21=ax2[:plot]([],[],color="seagreen",lw=4)[1]
		global line22=ax2[:plot]([],[],color="lightseagreen",lw=3)[1]

		ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
		ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
		SMALL_SIZE = 12
		MEDIUM_SIZE = 20
		BIGGER_SIZE = 30

		plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
		plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
		plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
		plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
		plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
		plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
		plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title






		# Define the init function, which draws the first frame (empty, in this case)
		function init()
		    global line1
		    global line2
		    global line3
			global line4
		    global line21
		    global line5
			global line22

			# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
			# global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

			line1[:set_data](data[indexrat][indexday][indextrial].trajectory[1,1],data[indexrat][indexday][indextrial].trajectory[1,2])
			line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat)
			line3[:set_data](xmaze,ymaze)
			line4[:set_data](data[indexrat][indexday][indextrial].trajectory[1,1],data[indexrat][indexday][indextrial].trajectory[1,2])
			line5[:set_data](data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2] .+ yplat)

			# global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

			# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
			# confidence=vcat(data[indexrat][indexday][indextrial].historyconfidence[1],[NaN for j=1:(size(data[indexrat][indexday][indextrial].trajectory,1)- 1)]);
			# line21[:set_data](collect(1:1:size(data[indexrat][indexday][indextrial].trajectory,1)),confidence)
			# line21[:set_data](collect(1:1:1),data[indexrat][indexday][indextrial].historyconfidence[1])
			line21[:set_data]([],[])
			line22[:set_data]([],[])	
			ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
			ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
			majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
			ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
			ax2.spines["top"].set_color("none")
			ax2.spines["right"].set_color("none")
			ax2.spines["bottom"].set_visible("False")
			ax2.spines["left"].set_visible("False")
			plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
			plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
			plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
			plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
			plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
			plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
			plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

		    return (line1,line2,line3,line4,line21,line5,line22, Union{})  # Union{} is the new word for None
		end



		function animate(i)
		    global line1
		    global line2
		    global line3
			global line4
		    global line21
		    global line5
			global line22

			i=i+1

			if i <= size(data[indexrat][indexday][indextrial].trajectory,1)
				# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
				# global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 

				line1[:set_data](data[indexrat][indexday][indextrial].trajectory[i,1],data[indexrat][indexday][indextrial].trajectory[i,2])
				line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial].indexstrategy] .+ yplat)
				line3[:set_data](xmaze,ymaze)
				line4[:set_data](data[indexrat][indexday][indextrial].trajectory[1:i,1],data[indexrat][indexday][indextrial].trajectory[1:i,2])
				line5[:set_data](data[indexrat][indexday][indextrial].real_platform[1].+xplat,data[indexrat][indexday][indextrial].real_platform[2].+ yplat)

				# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
				# global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

				x = (0:1:(i-1))
				line21[:set_data](x,data[indexrat][indexday][indextrial].historyconfidence[1:i])

				ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
				ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
				
				if i==size(data[indexrat][indexday][indextrial].trajectory,1) # mark the end of trial 1
					ylim=ax2.get_ylim()
					line22[:set_data]([x[size(data[indexrat][indexday][indextrial].trajectory,1)],x[size(data[indexrat][indexday][indextrial].trajectory,1)]],[ylim[1],ylim[2]]) 
				else 
					line22[:set_data]([],[]) # adjusting because the atan function ranges between -pi and pi and we want angles between 0 and 2pi
				end 
				
				majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
				ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
				ax2.spines["top"].set_color("none")
				ax2.spines["right"].set_color("none")
				ax2.spines["bottom"].set_visible("False")
				ax2.spines["left"].set_visible("False")
				plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
				plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
				plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
				plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
				plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
				plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
				plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
			else 
				index=i-size(data[indexrat][indexday][indextrial].trajectory,1);
				# global ax=subplot(get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 
				# global ax=fig3[:add_subplot](get(gs,(pycall(pybuiltin("slice"), PyObject, 1,4),pycall(pybuiltin("slice"), PyObject, 0,4)))) # we have to define it as a slice object bc 0:2 is not recognised 


				line1[:set_data](data[indexrat][indexday][indextrial2].trajectory[index,1],data[indexrat][indexday][indextrial2].trajectory[index,2])
				line2[:set_data](parameters[:Xplatform][data[indexrat][indexday][indextrial2].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][indextrial2].indexstrategy] .+ yplat)
				line3[:set_data](xmaze,ymaze)
				line4[:set_data](data[indexrat][indexday][indextrial2].trajectory[1:index,1],data[indexrat][indexday][indextrial2].trajectory[1:index,2])
				line5[:set_data](data[indexrat][indexday][indextrial2].real_platform[1].+xplat,data[indexrat][indexday][indextrial2].real_platform[2].+ yplat)

				# global ax2=subplot(get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))
				# global ax2=fig3[:add_subplot](get(gs,(0,pycall(pybuiltin("slice"), PyObject, 0,4))))

				x = (0:1:(i-1))
				line21[:set_data](x,vcat(data[indexrat][indexday][indextrial].historyconfidence[:],data[indexrat][indexday][indextrial2].historyconfidence[1:index]))
				ax2[:set_xlim]([0,size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)])
				ax2[:set_ylim]([minimum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence)),maximum(vcat(data[indexrat][indexday][indextrial].historyconfidence,data[indexrat][indexday][indextrial2].historyconfidence))])
				majors2 = [ floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)*l for l=0:floor( ( ( size(data[indexrat][indexday][indextrial].trajectory,1) + size(data[indexrat][indexday][indextrial2].trajectory,1) ) ) /floor((size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1))/6)) ]; # x locations of the ticks, we want 6 tickes max to look nice on the plot 
				ylim=ax2.get_ylim()
				line22[:set_data]([x[size(data[indexrat][indexday][indextrial].trajectory,1)],x[size(data[indexrat][indexday][indextrial].trajectory,1)]],[ylim[1],ylim[2]]) 
				ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors2)) 
				ax2.spines["top"].set_color("none")
				ax2.spines["right"].set_color("none")
				ax2.spines["bottom"].set_visible("False")
				ax2.spines["left"].set_visible("False")
				plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
				plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
				plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
				plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
				plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
				plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
				plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

			end 	
		    return (line1,line2,line3,line4,line21,line5,line22,Union{})  # Union{} is the new word for None
		end



		mywriter = anim.FFMpegWriter()
		myanim = anim.FuncAnimation(fig3, animate, init_func=init, frames=(size(data[indexrat][indexday][indextrial].trajectory,1)+size(data[indexrat][indexday][indextrial2].trajectory,1)), interval=200)
		myanim[:save]("Vid$(indexday)_$(indexrat).mp4",writer=mywriter) #bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

	end 
end 



