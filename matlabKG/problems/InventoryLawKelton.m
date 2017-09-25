% Simulates an (s,S) inventory policy for the problem from Section 1.5 of
% Simulation Modeling and Analysis, 4th edition, 2007, by Averill M. Law.
%
% The inputs are s and d, where d = S-s.
% Demand is 
function InventoryLawKelton(s,d,num_months, mean_interdemand, setup_cost,per_item_cost,holding_cost,backlog_cost)

setup_cost = 32;
per_item_cost = 3;
holding_cost = 1;
backlog_cost = 5;

total_ordering_costs = 0;
total_holding_cost = 0;
total_backlog_cost = 0;

sim_time = 0;
time_next_arrival = 
time_next_demand = sim_time + expon(mean_interdemand);
size_next_demand = 
time_next_month = 
time_end_sim = num_months;

time_of_next_event = zeros(1,4);
time_of_

while (1)  
  % Wait for the next event.  The possibilites are:
  % 1. Arrival of an order to the company from the supplier
  % 2. Demand for the product from a customer
  % 3. End of the simulation
  % 4. Inventory evaluation (and possible ordering) at the beginning of the
  % month

