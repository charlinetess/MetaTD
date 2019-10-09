

# Load the weights : 
cd("Documents/FosterDayanMorris/MetaRL") # got o dir 
#load packages 
using LinearAlgebra
using Statistics
using JLD2 
using FileIO

data_train=load("LearnWeights.jld2");
parameters=data_train["parameters"];
features=data_train["features"];
data=data_train["data"];

###################################################################################
###################################################################################
########################                                ###########################
########################       DEFINE FUNCTIONS         ###########################
########################                                ###########################
########################                                ##########################
########################                                ##########################
###################################################################################
###################################################################################

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
function reward(x,y,xp,yp,r) # x,y position of the rat and xp,yp position of the platform, r radius of the platform
    if (x-xp)^2+(y-yp)^2<= r^2 # if the rat is in the platform
        R=parameters[:rewardgoal];
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
#############################################################################
#############################################################################
#############################################################################
indexrat=1; 

indexstrategy=1; # Strategy the agent has chosen 
indexcurrentgoal=2; # real goal where the reward is 

estimated_platform_position=data[indexrat][indexstrategy].platformposition; # goal location corresponding to the chosen strategy 
estimated_actionmap=data[indexrat][indexstrategy].actionmap; # policy associated to this strategy 
estimated_valuemap=data[indexrat][indexstrategy].valuemap; # value map associated to this policy 

numberofstrategies=8;


currentxp=parameters[:Xplatform][indexcurrentgoal]; 
currentyp=parameters[:Yplatform][indexcurrentgoal];
estimatedxp=estimated_platform_position[1];
estimatedyp=estimated_platform_position[2];


global historyX=Float64[];
global historyY=Float64[];
#valuemap=Float64[];
global real_TDerrors=Float64[];
global estimated_TDerrors=[];


let k,t, timeout, prevdir,re,indexstart,currentposition
	# Initialise index to save the trajectory and the values 
	k=1;
	# initialise time 
	t=parameters[:times][k];      
	timeout=0;        
	prevdir=[0 0];   
	# Initialize reward 
	re=0;
	# Chose starting position :     
	indexstart=rand(1:4); # take indexstart-th starting position : chose     randomnly between 4 possibilities 1 East 2 North 3 West 4 South
	positionstart=[parameters[:Xstart][indexstart] parameters[:Ystart][indexstart]];
	currentposition=positionstart;


	while t<=parameters[:T] && re==0

			   if t==parameters[:T]
			       X=currentxp;
			       Y=currentyp;
			       currentposition=[X Y];
			       timeout=1; # if we have to put the rat on the platform     then we dont reinforce the actor but only the critic
			   end
			# Store former position to be able to draw trajectory
			push!(historyX,currentposition[1]); 
			push!(historyY,currentposition[2]);

			    # compute new activity of pace cells :
			# actplacecell=place_activity(   position[1],position[2],Xplacecell,Yplacecell,σ); # this    function is wrong 
			actplacecell=placecells([currentposition[1],currentposition[2]],   parameters[:centres],parameters[:σPC]);

			### Compute Critic ###
			critic=[(transpose(actplacecell)*data[indexrat][indexstrategy].valuemap)[1] for indexstrategy=1:numberofstrategies]; # current estimation of the future discounted reward - computed for every strategy 
			C=dot(estimated_valuemap,actplacecell); # compute current value using the current belief 

			####### Take decision and move to new position : ########
			#  Compute action cell activity    
			actactioncell=transpose(estimated_actionmap)*actplacecell; # careful z    contains place cells in rows and action cells in column 
			   if maximum(actactioncell)>=100
			       actactioncell=100*actactioncell./maximum(actactioncell); 
			   end
			# Compute probability distribution : 
			Pactioncell=exp.(parameters[:temperature]*actactioncell)./sum(exp.(parameters[:temperature]*actactioncell)); 
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
			       global dir
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
			real_re=reward(currentposition[1],currentposition[2],currentxp,currentyp,parameters[:r]); # compute the reward that tge agent actually gets  
			estimated_re=reward(currentposition[1],currentposition[2],estimatedxp,estimatedyp,parameters[:r]); # computes the reward the agent believes it is getting 


			actplacecell=placecells([currentposition[1],currentposition[2]],   parameters[:centres],parameters[:σPC]);

			   if re==1 # if we are on the platform 
			      ###  Compute error ###
			       C_next=0;
			       critic_next=zeros(numberofstrategies)
			   else 
			       C_next=dot(estimated_valuemap,actplacecell);# new estimation of the future discounted reward in current belief 
			       critic_next=[(transpose(actplacecell)*data[indexrat][indexstrategy].valuemap)[1] for indexstrategy=1:numberofstrategies]; # current estimation of the future discounted reward - computed for every strategy 
			   end 


			#### Compute errors  ####
			estimated_errors=estimated_re*ones(numberofstrategies,1).+parameters[:γ]*critic_next[:].-critic[:]; 


			real_err=real_re+parameters[:γ]*C_next-C[1]; 

			push!(real_TDerrors,real_err);
			push!(estimated_TDerrors,estimated_errors[:])

			k=k+1; # counting steps
			t=parameters[:times][k]; # counting time
			##################################################            
	end # end trial 
end # end scope t, k , etc..
