

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
numberoftrials=features[:numberoftrials];
numberofrats=features[:numberofrats];
numberofdays=features[:numberofdays];
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


function sigmoid(c)
    return ρ*inv.(ones(length(c)).+exp.(-β*(c.-h*ones(length(c)))))
end

function confidencedynamics(confidence,SPE)
	newconfidence=(1-parameters[:dt]/τconfidence)*confidence+SPE;
	return newconfidence
end

#############################################################################
#############################################################################
#############################################################################
indexrat=1; 




numberofstrategies=8;


# parameters for actor's activation function: 
β=1; # define the steepness of the function. the highest the closest to heaviside 
h=-0.3; # define the cut. Threshold of activation to be kept
# gain factor 
ρ=2;

τconfidence=15; # tau need to be suuuuuuuuper long so that we can  
# check sigmoid parameters around the values of confidence we will get, perfect 
# using PyPlot
# plot(-1.1:0.01:1.1,sigmoid(-1.1:0.01:1.1))
# show()


numberoftrialstest=4;
numberofdaystest=2;

global historyX=Float64[];
global historyY=Float64[];
#valuemap=Float64[];
global real_TDerrors=Float64[];
global estimated_TDerrors=[];
global historyconfidence=[];
global historySPE=[];
global historytemperature=[];

global temperature=2; # we start with 2 and will adjust this as a function of the confidence 

currentexperiment=[]; # TO CHANGE WHEN CONSIDERING MULTIPLE RUNS 

for indexday=1:numberofdaystest
	println("indexday$(indexday)")
let real_err=0,estimated_errors=zeros(numberofstrategies),currentxp,currentyp, currentday
	indexcurrentgoal=rand(1:numberofstrategies); # real goal where the reward is 
	currentxp=parameters[:Xplatform][indexcurrentgoal]; 
	currentyp=parameters[:Yplatform][indexcurrentgoal];

	currentday=[];

	for indextrial=1:numberoftrialstest # at every trial we define the new strategy that the rat will follow 
		println("indextrial$(indextrial)")
		let indexstrategy, estimated_actionmap, estimated_valuemap, k,t, timeout, prevdir,re,indexstart,currentposition,confidence=2, temperature=sigmoid(confidence)[1]; # at start of a trial the agent is confident about the strategy to follow 
			# chose strategy 
			if  !(length(findall((estimated_errors.-real_err*ones(length(estimated_errors))).==minimum(estimated_errors.-real_err*ones(length(estimated_errors)))))==1)  # if there are more than one potential favourite strategies, we chose randomly among them 
				indexstrategy=rand(findall((estimated_errors.-real_err*ones(length(estimated_errors))).==minimum(estimated_errors.-real_err*ones(length(estimated_errors)))))
			else
				indexstrategy=argmin(estimated_errors.-real_err*ones(length(estimated_errors)));
			end
			estimated_actionmap=data[indexrat][indexstrategy].actionmap; # policy associated to this strategy 
			estimated_valuemap=data[indexrat][indexstrategy].valuemap; # value map associated to this policy 

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


			while t<=parameters[:T] && real_re==0

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

					# Compute temperature out of confidence :
					temperature=sigmoid(confidence)[1];
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
					estimated_re=[reward(currentposition[1],currentposition[2],data[indexrat][indexstrategy].platformposition[1],data[indexrat][indexstrategy].platformposition[2],parameters[:r]) for indexstrategy=1:numberofstrategies]; # computes the reward the agent believes it is getting 


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
					estimated_errors=estimated_re.+parameters[:γ]*critic_next[:].-critic[:]; 

					real_err=real_re+parameters[:γ]*C_next-C[1]; 


					# update confidence based on this strategy prediction error  : 
					SPE=real_err-estimated_errors[indexstrategy];  # strategy prediction error 
					confidence=confidencedynamics(confidence,SPE) # confidence gets a shot of -1 when it thinks it is on the goal but does not receive reward, 0 when nothing surprising happens and 1 when it finds the goal without having planned that this is where the goal is
					push!(historySPE,SPE)
					push!(historyconfidence,confidence)
					push!(real_TDerrors,real_err);
					push!(estimated_TDerrors,estimated_errors[:])
					k=k+1; # counting steps
					t=parameters[:times][k]; # counting time
					##################################################            
			end # end trial 
			currenttrial=(trajectory=hcat(historyX,historyY),historySPE=historySPE,latency=t,historyconfidence=historyconfidence,real_TDerrors=real_TDerrors,estimated_TDerrors=estimated_TDerrors,historytemperature=historytemperature,real_platform=[currentxp,currentyp],indexstrategy=indexstrategy); # Creating the current trial with all    its fields
            push!(currentday,currenttrial);
		end # end scope t, k , etc..
	end # end loop over trials 
	push!(currentexperiment,currentday)
end # end scope of error, etc..

end # end loop over days 


################ now plot the TD errors so we can have an idea of the range 

using PyPlot 
figure()
for k=3#numberofstrategies
	plot([estimated_TDerrors[i][k] for i=1:length(estimated_TDerrors)])
end
plot(real_TDerrors)
show()



theta=0:pi/50:(2*pi+pi/50); # to plot circles 
# plot trajectory 
figure()
plot(parameters[:R]*cos.(theta),parameters[:R]*sin.(theta),"k-")
for k=1:numberofstrategies
	plot(data[indexrat][k].platformposition[1].+parameters[:r]*cos.(theta),data[indexrat][k].platformposition[2].+parameters[:r]*sin.(theta),"m-")
end
plot(historyX,historyY,"b")
show()

# chose day 
indexday=1;
indextrial=1;

# plot confidence evolution 
subplot(5,1,1)
plot(currentexperiment[indexday][indextrial].historyconfidence)
subplot(5,1,2)
plot(currentexperiment[indexday][indextrial].historytemperature)
subplot(5,1,3)
plot(currentexperiment[indexday][indextrial].historySPE)
subplot(5,1,4)
plot(currentexperiment[indexday][indextrial].real_TDerrors)
subplot(5,1,5)
plot(currentexperiment[indexday][indextrial].estimated_TDerrors[currentexperiment[indexday][indextrial].indexstrategy])
show()


using PyPlot 
figure()
for k=1:numberofstrategies
	subplot(numberofstrategies+1,1,k)
	plot([estimated_TDerrors[i][k] for i=1:length(estimated_TDerrors)])
end
subplot(numberofstrategies+1,1,numberofstrategies+1)
plot(real_TDerrors)
show()



using PyPlot
for k=1:numberofdaystest
# Calculate standard deviation 
#err=[std([rats.experiment[n].day[k].trial[i].Latency for n in 1:numberofrats]; corrected=false) for i in 1:numberoftrials] ;

# Calculate the lower value for the error bar : 
#uppererror = [std([data[n][k].currentplatform[i].latency for n in 1:numberofrats]; corrected=false)./sqrt(numberofrats) for i in 1:numberoftrials] ;
#lowererror = [std([data[n][k].currentplatform[i].latency for n in 1:numberofrats]; corrected=false)./sqrt(numberofrats) for i in 1:numberoftrials] ;

#errs=[lowererror,uppererror];

PyPlot.plot(k*numberoftrialstest.+(0:numberoftrialstest-1), [currentexperiment[k][i].latency for i in 1:numberoftrialstest ], marker="None",linestyle="-",color="darkgreen",label="Base Plot")
  

#PyPlot.errorbar(k*numberoftrials.+(0:numberoftrials-1),[mean([data[n][k].currentplatform[i].latency for n in 1:numberofrats]) for i in 1:numberoftrials],yerr=errs,fmt="o",color="k")
end













