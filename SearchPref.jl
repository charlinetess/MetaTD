# This file generates the search preference on the second trial of the same goal location in trials in which the goal is missing 


#load packages 
using LinearAlgebra
using Statistics
using JLD2 
using FileIO

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


function sigmoid(c,param)
    return param[:ρ]*inv.(ones(length(c)).+exp.(-param[:β]*(c.-param[:h]*ones(length(c)))))
end

function confidencedynamics(confidence,SPE,meta_parameters)
	newconfidence=(1-meta_parameters[:dt]/meta_parameters[:τconfidence])*confidence+SPE;
	return newconfidence
end


radiussearchpref=20; # radius of the area in which we calculate searchpreference 


numberofstrategies=8;

# parameters for  the sigmoid defining the link between confidence and exploration 
β=8; # define the steepness of the function. the highest the closest to heaviside 
h=-0.2; # define the cut. Threshold of activation to be kept
h=0.0; # define the cut. Threshold of activation to be kept

# gain factor 
ρ=2;

τconfidence=400; # tau need to be suuuuuuuuper long so that we can  
# check sigmoid parameters around the values of confidence we will get, perfect 
# using PyPlot
# plot(-1.1:0.01:1.1,sigmoid(-1.1:0.01:1.1))
# show()


numberoftrialstest=4;
numberofdaystest=8;
Tprobetrials=60;

meta_parameters=Dict(:numberofstrategies=>numberofstrategies,:β=>β,:h=>h,:ρ=>ρ,:τconfidence=>τconfidence,:numberoftrialstest=>numberoftrialstest,:numberofdaystest=>numberofdaystest);
meta_parameters[:dt]=parameters[:dt];


# Initialise index to save the trajectory and the values 
k=1;
# initialise time 
t=parameters[:times][k];      
timeout=0;        
prevdir=[0 0];   
# Initialize reward 
re=0;
real_re=0;

global searchinzones=0;
global searchpref=0;

global SearchPref=[];


for indexrat=1:features[:numberofrats]
	# Initialise index to save the trajectory and the values 
	println(indexrat)
	k=1;
	# initialise time 
	t=parameters[:times][k];      
	timeout=0;        
	prevdir=[0 0];   
	historyX=Float64[];
	historyY=Float64[];
	real_TDerrors=Float64[];
	estimated_TDerrors=[];
	historyconfidence=[];
	historySPE=[];
	historytemperature=[];
	deception=0; 


	real_err=0;
	estimated_errors=zeros(meta_parameters[:numberofstrategies]);
	SPE=zeros(meta_parameters[:numberofstrategies]);
	real_re=0;
	global confidence
	confidence=0.5;
	temperature=sigmoid(confidence,meta_parameters)[1];

	# Initialize reward 
	re=0;
	real_re=0;
	global searchinzones=0;
	global searchpref=0;

	temperature=sigmoid(confidence,meta_parameters)[1];
	historyX=Float64[];
	historyY=Float64[];
	real_TDerrors=Float64[];
	estimated_TDerrors=[];
	historyconfidence=[];
	historySPE=[];
	historytemperature=[];
	deception=0; 



	indexstrategy=rand(1:meta_parameters[:numberofstrategies])
	actionmap=data[indexrat][indexstrategy].actionmap; # policy associated to this strategy 
	valuemap=data[indexrat][indexstrategy].valuemap; # value map associated to this policy 

	currentxp=data[indexrat][indexstrategy].platformposition[1];
	currentyp=data[indexrat][indexstrategy].platformposition[2];


	# Chose starting position :     
	indexstart=rand(1:4); # take indexstart-th starting position : chose     randomnly between 4 possibilities 1 East 2 North 3 West 4 South
	positionstart=[parameters[:Xstart][indexstart] parameters[:Ystart][indexstart]];

	currentposition=positionstart;

	while t<=Tprobetrials # we put no learning on actorweights and criticweights on probe trials
	            # if t==Tprobetrials
	            #     X=currentxp;
	            #     Y=currentyp;
	            #     currentposition=[X Y];
	            #     timeout=1; # if we have to put the rat on the platform then we dont reinforce the actor but only the critic
	            #     platform=1;
	            # end                     
	        push!(historyX,currentposition[1]); 
	        push!(historyY,currentposition[2]);
	        # compute new activity of pace cells :
	        actplacecell=placecells([currentposition[1],currentposition[2]],parameters[:centres],parameters[:σPC]);  
	            if !(k==1)
	                formeractplacecell=actplacecell; # need storing to compute the self motion estimate
	            end       

			critic=[dot(data[indexrat][indexstrategy].valuemap,actplacecell) for indexstrategy=1:meta_parameters[:numberofstrategies]]; # current estimation of the future discounted reward - computed for every strategy                
	        ### Compute Critic ###
	        # C=dot(criticweights,actplacecell); # current estimation of the future discounted reward         

	        ####### Take decision and move to new position : ######## 
	        actactioncell=transpose(actionmap)*actplacecell; # Compute action cells activity actorweights contains place cells in rows and action cells in column 
	            if maximum(abs.(actactioncell))>=100
	                actactioncell=100*actactioncell./maximum(abs.(actactioncell)); 
	            end

		# Compute temperature out of confidence :
		temperature=sigmoid(confidence,meta_parameters)[1];
		push!(historytemperature,temperature )
		# Compute probability distribution :
		Pactioncell=exp.(temperature*actactioncell)./sum(exp.(temperature*actactioncell)); 
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
		       dir
		       dir=dir./norm(dir); # normalize so we control the exact    speed of the rat
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
		# real_re=reward(currentposition[1],currentposition[2],currentxp,currentyp,parameters); # compute the reward that tge agent actually gets  
		real_re=0; # here we mimick the removal of the platform 
		estimated_re=[reward(currentposition[1],currentposition[2],data[indexrat][indexstrategy].platformposition[1],data[indexrat][indexstrategy].platformposition[2],parameters) for indexstrategy=1:meta_parameters[:numberofstrategies]]; # computes the reward the agent believes it is getting 



		actplacecell=placecells([currentposition[1],currentposition[2]],   parameters[:centres],parameters[:σPC]);

		   if real_re==1 # if we are on the platform 
		      ###  Compute error ###
		       # C_next=0;
		       critic_next=zeros(meta_parameters[:numberofstrategies])
		   else 
		       # C_next=dot(estimated_valuemap,actplacecell);# new estimation of the future discounted reward in current belief 
		       # critic_next=[(transpose(actplacecell)*data[indexrat][indexstrategy].valuemap)[1] for indexstrategy=1:meta_parameters[:numberofstrategies]]; # current estimation of the future discounted reward - computed for every strategy 
		       critic_next=[dot(data[indexrat][indexstrategy].valuemap,actplacecell) for indexstrategy=1:meta_parameters[:numberofstrategies]]; 
		   end 


		#### Compute errors  ####
		# estimated_errors=real_re.+parameters[:γ]*critic_next[:].-critic[:]; 

		estimated_errors=estimated_re.+parameters[:γ]*critic_next[:].-critic[:]; 

		real_err=real_re+parameters[:γ]*critic_next[indexstrategy]-critic[indexstrategy]; 

		# real_err=real_re+parameters[:γ]*C_next-C[1]; 
		#println(real_re)

		# update confidence based on this strategy prediction error  : 
		#global SPE=real_err-estimated_errors[indexstrategy];  # strategy prediction error 
		#global SPE=critic[:].-real_re;  # strategy prediction error 

		SPE=real_err.-estimated_errors; 
		println(SPE)
		println(confidence)
		#SPE=C-real_re;


		if (deception==0)&(SPE[indexstrategy]<0)
			confidence=confidencedynamics(confidence,SPE[indexstrategy],meta_parameters) # confidence gets a shot of -1 when it thinks it is on the goal but does not receive reward, 0 when nothing surprising happens and 1 when it finds the goal without having planned that this is where the goal is
		elseif (deception==1)&(SPE[indexstrategy]<0) # if we already have been disappointed but the SPE is still negative 
			confidence=confidencedynamics(confidence,0,meta_parameters) # confidence gets a shot of -1 when it thinks it is on the goal but does not receive reward, 0 when nothing surprising happens and 1 when it finds the goal without having planned that this is where the goal is
		else
			confidence=confidencedynamics(confidence,SPE[indexstrategy],meta_parameters)
		end 

		if SPE[indexstrategy]<0 # the first encounter with a negative strategy prediction error reduces the confidence and we dont reduce it afterwards
			deception=1
		end 

		push!(historySPE,SPE)
		push!(historyconfidence,confidence)
		push!(real_TDerrors,real_err);
		push!(estimated_TDerrors,estimated_errors[:])
		
		k=k+1; # counting steps
		
		t=parameters[:times][k]; # counting time
	            ####### ####### ####### Updating search preference  ####### ####### #######
	            if (currentposition[1]-currentxp)^2+(currentposition[2]-currentyp)^2<= radiussearchpref^2 
		            global searchpref  
	                searchpref=searchpref+1*parameters[:dt];
	                # println(searchpref)
	            end
	            # compute time in the 8 zones :
	            if ((currentposition[1]-parameters[:Xplatform][1])^2+(currentposition[2]-parameters[:Yplatform][1])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][2])^2+(currentposition[2]-parameters[:Yplatform][2])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][3])^2+(currentposition[2]-parameters[:Yplatform][3])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][4])^2+(currentposition[2]-parameters[:Yplatform][4])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][5])^2+(currentposition[2]-parameters[:Yplatform][5])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][6])^2+(currentposition[2]-parameters[:Yplatform][6])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][7])^2+(currentposition[2]-parameters[:Yplatform][7])^2<= radiussearchpref^2)|((currentposition[1]-parameters[:Xplatform][8])^2+(currentposition[2]-parameters[:Yplatform][8])^2<=radiussearchpref^2)
	            	global searchinzones
	                searchinzones=searchinzones+1*parameters[:dt];
	            end


	end # end while 

	global SearchPref
	global searchpref,searchinzones
	if searchinzones==0
		push!(SearchPref,searchpref)
	else 
		push!(SearchPref,searchpref/searchinzones)
	end 

end

# 	
# 	           ,,
# 	         `7MM           mm
# 	           MM           MM
# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm
# 	  MM   `Wb MM 6W'   `Wb MM
# 	  MM    M8 MM 8M     M8 MM
# 	  MM   ,AP MM YA.   ,A9 MM
# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo
# 	  MM
# 	.JMML.


searchprefs=mean(SearchPref); # compute mean, the probe trial is always trial2
# plot bar plot 
using PyPlot
ioff()
fig = figure("Test plot search Preference",figsize=(4,9))
ax = gca()

SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

uppererror =std(SearchPref*100)./sqrt(features[:numberofrats]); 
lowererror = std(SearchPref*100)./sqrt(features[:numberofrats]);
errs=[lowererror  uppererror]';

ax.bar(1,mean(SearchPref*100),width=0.1,yerr=errs,color=[60/255,179/255,113/255],align="center",alpha=0.4)
ax[:axes][:get_xaxis]()[:set_ticks]([])
#xlabel=["day $(indexprobedays[1])","day $(indexprobedays[2])","day $(indexprobedays[3])"]
axis("tight")



ax.plot(0:1:2,12.5*ones(size(0:1:2,1)),color="green",label="Chance",linestyle="--")

locs, labels = xticks()  
# xticks(1:1:length(parameters[:indexprobedays]), ["day $(parameters[:indexprobedays][1])","day $(parameters[:indexprobedays][2])","day $(parameters[:indexprobedays][3])"])

ax[:grid](false);
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
legend()
# title("%time spent in the correct zone")

show()
