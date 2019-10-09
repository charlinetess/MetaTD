

# Load the weights : 
cd("Documents/FosterDayanMorris/MetaRL") # got o dir 
using JLD2 
using FileIO

data_train=load("LearnWeights.jld2");
parameters=data_train["parameters"];
features=data_train["features"];
data=data_train["data"];

indexrat=1;
indexstrategy=1;
indexcurrentgoal=2;

estimated_platform_position=data[indexrat][indexstrategy].platformposition;
estimated_actionmap=data[indexrat][indexstrategy].actionmap;
estimated_valuemap=data[indexrat][indexstrategy].valuemap;

numberofstrategies=8;



currentxp=parameters[:Xplatform][indexcurrentgoal];
currentyp=parameters[:Yplatform][indexcurrentgoal];
estimatedxp=estimated_platform_position[1];
estimatedyp=estimated_platform_position[2];

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
   critic=[dot(data[indexrat][indexstrategy].actionmap,actplacecell) for indexstrategy=1:numberofstrategies]; # current estimation of the future discounted reward - computed for everystrategy 
   
   ####### Take decision and move to new position : ########
   # Compute the activity of action cells 

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
   dir=(newdir./(1.0+momentum).+momentum.*prevdir./(1.0+momentum));     # smooth trajectory to avoid sharp angles
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
   real_re=reward(currentposition[1],currentposition[2],currentxp,currentyp,parameters[:r]); 
   estimated_re=reward(currentposition[1],currentposition[2],currentxp,currentyp,parameters[:r]); 


   actplacecell=placecells([currentposition[1],currentposition[2]],   parameters[:centres],parameters[:σPC]);

       if re==1 # if we are on the platform 
          ###  Compute error ###
           Cnext=0;
       else 
           Cnext=dot(estimated_valuemap,actplacecell);# new estimation     of the future discounted reward 
       end 
   

   #### Compute errors  ####

   estimated_err=estimated_re+parameters[:γ]*Cnext-C[1]; 


   real_err=real_re+parameters[:γ]*Cnext-C[1]; 

   push!(real_TDerrors,real_err);
   push!(estimated_TDerrors,)

   k=k+1; # counting steps
   t=parameters[:times][k]; # counting time
##################################################            
end
