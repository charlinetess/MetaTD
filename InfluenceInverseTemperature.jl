# This file generates 3 trajectories from the weights trained using the actor critic architecture, for 3 different values of the inverse temperature beta. 

#load packages 
using LinearAlgebra
using Statistics
using JLD2 
using FileIO
using PyPlot
using PyCall



data_train=load("LearnWeights.jld2");
parameters=data_train["parameters"];
features=data_train["features"];
data=data_train["data"];
numberoftrials=features[:numberoftrials];
numberofrats=features[:numberofrats];
numberofdays=features[:numberofdays];


# 	
# 	                                                 ,,
# 	`7MM"""YMM                                mm     db
# 	  MM    `7                                MM
# 	  MM   d `7MM  `7MM  `7MMpMMMb.  ,p6"bo mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.  ,pP"Ybd
# 	  MM""MM   MM    MM    MM    MM 6M'  OO   MM     MM 6W'   `Wb MM    MM  8I   `"
# 	  MM   Y   MM    MM    MM    MM 8M        MM     MM 8M     M8 MM    MM  `YMMMa.
# 	  MM       MM    MM    MM    MM YM.    ,  MM     MM YA.   ,A9 MM    MM  L.   I8
# 	.JMML.     `Mbod"YML..JMML  JMML.YMbmd'   `Mbmo.JMML.`Ybmd9'.JMML  JMML.M9mmmP'
# 	
# 	
# Compute the activity of the place cells
function  placecells(pos,centres,width)
	#
	# PLACECELLS(POSITION,CENTRES,WIDTH) calculates the activity of the place cells
	#in the simulation. The returned vector F is of length N, where N is the number of place
	#cells, and it contains the activity of each place cell given the simulated rat's current
	#POSITION (a 2 element column vector). The activity of the place cells is modelled as a
	#rate-of-fire (i.e. a scalar value) determined by a gaussian function. The CENTRES of the
	#gaussian functions are an argument, and must be a 2 x N matrix containing each place
	#cell's preferred location in 2D space. The WIDTH of the place cell fields must
	#also be provided as a scalar value (all place cells are assumed to have the same
	#width).
	#
	#The returned vector, F, must be a N element column vector.
	    # calculate place cell activity
	F = exp.(-sum((repeat(pos,1,size(centres,2))-centres).^2,dims=1)/(2*width^2))';
	return F
end



# Calculate reward as a function of position 
function reward(x,y,xp,yp,param) # x,y position of the rat and xp,yp position of the platform, r radius of the platform
    if (x-xp)^2+(y-yp)^2<= param[:r]^2 # if the rat is in the platform
        R=param[:rewardgoal];
    else # else 
        R=0;
    end 
end


# This function tells within wich index column is located x, used to take decision on which action to follow
function indice(Acum,x) # x number, Acum vector
    
    for i=1:length(Acum)
       if i==1
           if x<Acum[i] # if the random number generated is before the first 
                return i
            end
        else
            if Acum[i-1]<x<=Acum[i]
                return i
            end
        end
    end  
        
end


temperature=2;

# Initialise index to save the trajectory and the values 
k=1;
# initialise time 
t=parameters[:times][k];      
timeout=0;        
prevdir=[0 0];   
# Initialize reward 
re=0;
real_re=0;
# Chose starting position :     
indexstart=rand(1:4); # take indexstart-th starting position : chose     randomnly between 4 possibilities 1 East 2 North 3 West 4 South
positionstart=[parameters[:Xstart][indexstart] parameters[:Ystart][indexstart]];
currentposition=positionstart;



indexrat=1;
indexstrategy=1; 
actionmap=data[indexrat][indexstrategy].actionmap; # policy associated to this strategy 
# valuemap=data[indexrat][indexstrategy].valuemap; # value map associated to this policy 
xp=data[indexrat][indexstrategy].platformposition[1];
yp=data[indexrat][indexstrategy].platformposition[2];

global historyX=[];
global historyY=[];

actplacecell=placecells([currentposition[1],currentposition[2]],   parameters[:centres],parameters[:σPC]);
####### Take decision and move to new position : ########
#  Compute action cell activity    
actactioncell=transpose(actionmap)*actplacecell; # careful z    contains place cells in rows and action cells in 
Pactioncell=exp.(temperature*actactioncell)./sum(exp.(temperature*actactioncell)); 

global historyPactioncell=Pactioncell;

while t<=parameters[:T] && real_re==0

		   if t==parameters[:T]
		       X=xp;
		       Y=yp;
		       currentposition=[X Y];
		       timeout=1; # if we have to put the rat on the platform     then we dont reinforce the actor but only the critic
		   end
		# Store former position to be able to draw trajectory
		push!(historyX,currentposition[1]); 
		push!(historyY,currentposition[2]);
		# compute new activity of pace cells :
		# actplacecell=place_activity(   position[1],position[2],Xplacecell,Yplacecell,σ); # this    function is wrong 
		actplacecell=placecells([currentposition[1],currentposition[2]],   parameters[:centres],parameters[:σPC]);
		####### Take decision and move to new position : ########
		#  Compute action cell activity    
		actactioncell=transpose(actionmap)*actplacecell; # careful z    contains place cells in rows and action cells in column 
		   if maximum(actactioncell)>=100
		       actactioncell=100*actactioncell./maximum(actactioncell); 
		   end
		# Compute probability distribution :
		Pactioncell=exp.(temperature*actactioncell)./sum(exp.(temperature*actactioncell)); 
		historyPactioncell=hcat(historyPactioncell,Pactioncell)
		# Compute summed probability distribution:
		#SumPactioncell=cumul(Pactioncell);
		SumPactioncell=[sum(Pactioncell[1:k]) for k=1:length(Pactioncell)    ]
		# Compute summed probability distribution:
		# SumPactioncell=cumul(Pactioncell); # other possibility 
		# Generate uniform number between 0 and 1 :
		x=rand();
		# now chose action: 
		indexaction=indice(SumPactioncell,x); # Chose which action     between the 8 possibilities
		argdecision=parameters[:angles][indexaction]; # compute the coreesponding     angle 
		newdir=[cos(argdecision) sin(argdecision)];
		dir=(newdir./(1.0+parameters[:momentum]).+parameters[:momentum].*prevdir./(1.0+parameters[:momentum]));     # smooth trajectory to avoid sharp angles
		   if !(norm(dir)==0)
		       dir=dir./norm(dir); # normalize so we control the exact speed of the rat
		   end
		formerposition=currentposition;
		# Compute new position : 
		currentposition=currentposition.+parameters[:dt].*parameters[:speed].*dir; 
		if currentposition[1]^2+currentposition[2]^2>=parameters[:R]^2 # if we are outside of circle, move a lil bit 
		   currentposition = (currentposition./norm(currentposition))*(parameters[:R]     - parameters[:R]/50);
		   if !(norm(currentposition-formerposition)==0)
		       dir=(currentposition-formerposition)./norm(    currentposition-formerposition);
		   else
		       dir=[0 0]
		   end
		end
		prevdir=dir;                           
		# compute new activity of pace cells :
		# actplacecell=place_activity(   position[1],position[2],Xplacecell,Yplacecell,σ);
		  ###  Compute reward ### 
		real_re=reward(currentposition[1],currentposition[2],xp,yp,parameters); # compute the reward that tge agent actually gets  		
		k=k+1; # counting steps
		t=parameters[:times][k]; # counting time
		##################################################            
end # end trial 

# plot trajectory historyX,historyY
push!(historyX,currentposition[1]); 
push!(historyY,currentposition[2]);

argument=0:pi/50:2pi+pi/50;
xplat=parameters[:r]*cos.(argument);
yplat=parameters[:r]*sin.(argument);
xmaze=parameters[:R]*cos.(argument);
ymaze=parameters[:R]*sin.(argument);
fig=figure()
ax=fig.gca()

ax[:set_ylim]([-101,101])
ax[:set_xlim]([-101,101])
plot(xp.+xplat,yp.+ yplat,color="darkgoldenrod",lw=2) # current goal 
# plot(parameters[:Xplatform][data[indexrat][indexday][1].indexstrategy].+xplat,parameters[:Yplatform][data[indexrat][indexday][1].indexstrategy] .+ yplat,color="darkgrey",lw=1) # goal of current strategy 
plot(xmaze,ymaze,color="darkslategray", lw=2) # maze boundaries 
# plot(data[indexrat][indexday][2].trajectory[end,1],data[indexrat][indexday][2].trajectory[end,2],color="teal","*", markersize=12) # agent 
plot(historyX[:],historyY[:],color="slateblue",lw=2) # trajectory 


ax.set_axis_off()

show()

# meanPactioncell=[ mean([historyPactioncell[k][i] for k=1:length(historyPactioncell)]) for i=1:parameters[:NA]]

fig=figure()
ax=fig.gca()
bar(1:1:parameters[:NA],mean([historyPactioncell[:,k] for k=1:size(historyPactioncell,2)]),color="seagreen")#color=[74,181,57]./255)
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["bottom"].set_visible("False")
ax.spines["left"].set_visible("False")
labels=[" ", "SW","S","SE","E","NE","N","NW","W"]
ax.set_xticklabels(labels)
ax.set_xlabel("Direction")#,size=18)
ax.set_ylabel("Mean probability during trial")#,size=18)

show()


