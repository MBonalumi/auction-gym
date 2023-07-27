import time
start_time = time.time()
ts = time.strftime("%Y%m%d-%H%M", time.localtime())
folder_name = None

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from main import parse_config, instantiate_agents, instantiate_auction, simulation_run
from tqdm import tqdm
from utils import get_project_root
from pathlib import Path
import concurrent.futures

import ray

# CONSTANTS
ROOT_DIR = get_project_root()

# INDEXES of the return
idx_auction_rev = 0
idx_social_welfare = 1
idx_advertisers_surplus = 2
idx_cumulative_surpluses = 3
idx_instant_surpluses = 4
idx_regrets = 5
idx_actions_rewards = 6

# LOGS
agents_update_logs = []
# contexts_logs = []
# agents_bids_logs = []


def agent_update(agent, iteration):
    start = time.time()

    if len(agent.logs) > 0:
        agent.update(iteration=iteration)
        agent.clear_utility()
        agent.clear_logs()

    agents_update_logs.append(f'iteration: {iteration:4.0f}, agent: {agent.name:>15}, duration: {time.time() - start:3.4f}')
    return

@ray.remote
def run_repeated_auctions(num_run, num_runs, instantiate_agents_args, instantiate_auction_args, results=None, debug=False):

    rng, agent_configs, agents2item_values, agents2items = instantiate_agents_args
    config, max_slots,embedding_size,embedding_var,obs_embedding_size = instantiate_auction_args

    # Placeholders for output
    auction_revenue = []
    social_welfare = []
    advertisers_surplus = []

    contexts_logs = []
    agents_bids_logs = []
    
    # Instantiate Agent and Auction objects
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    
    agents_overall_surplus = [[] for _ in range(len(agents))]

    agents_instant_surplus = [[] for _ in range(len(agents))]

    agents_regret_history = [[] for _ in range(len(agents))] #TODO
    agents_actionsrewards_history = [[] for _ in range(len(agents))] #TODO

    ### SECONDARY OUTPUTS ###
    agents_last_avg_utilities = [[] for _ in range(len(agents))]

    # Instantiate Auction object
    auction, num_iter, rounds_per_iter, output_dir = \
        instantiate_auction(rng,
                            config,
                            agents2items, agents2item_values, agents,
                            max_slots,
                            embedding_size, embedding_var, obs_embedding_size)
    
    # give bidder info about the auction type (2nd price, 1st price, etc.)
    # to calculate REGRET IN HINDISGHT
    from BidderBandits import BaseBandit
    for i, agent in enumerate(auction.agents):
        if isinstance(agent.bidder, BaseBandit):
            agent.bidder.auction_type = config['allocation']
            agent.bidder.agent_id = i
            agent.bidder.num_iterations = num_iter
            if num_run == 0: 
                if not agent.bidder.isContinuous:
                    print('\t', agent.name, ': ', agent.bidder.BIDS)
                else:
                    print('\t', agent.name, ': ', agent.bidder.textContinuous)

    if debug:
        for agent in auction.agents:
            print(agent.name, ': ', agent.bidder.auction_type, end=' | ')

    # Run repeated auctions
    # This logic is encoded in the `simulation_run()` method in main.py
    # print(num_run, ') ', end='')
    # from tqdm import tqdm
    # with tqdm(total=num_iter, desc=f'{num_run+1}/{num_runs}', leave=True) as pbar:
    for i in range(num_iter):
        if debug: print(f'Iteration {i+1} of {num_iter}')

        # Simulate impression opportunities
        opportunities_results = []  
        for _ in range(rounds_per_iter):
            opportunities_results.append( auction.simulate_opportunity() )
        
        participating_agents_ids = np.array(np.array(opportunities_results)[:,0,:], dtype=np.int32)
        iter_bids = np.array(np.array(opportunities_results)[:,1,:], dtype=np.float32)
        
        participating_agents_masks = [np.isin(participating_agents_ids, agent).any(axis=1) for agent in range(len(agents))]

        sorted_bids_iter = np.sort(iter_bids, axis=1)
        maximum_bids_iter = sorted_bids_iter[:,-1]
        second_maximum_bids_iter = sorted_bids_iter[:,-2]

        # Log 'Gross utility' or welfare
        social_welfare.append(sum([agent.gross_utility for agent in auction.agents]))

        # Log 'Net utility' or surplus
        advertisers_surplus.append(sum([agent.net_utility for agent in auction.agents]))
        for agent_id, agent in enumerate(auction.agents):
            #surplus
            agents_instant_surplus[agent_id].append(agent.net_utility)
            agents_overall_surplus[agent_id].append(np.array(agents_instant_surplus[agent_id], dtype=object).sum())
            
            # winning bids
            agent.bidder.winning_bids = maximum_bids_iter
            agent.bidder.second_winning_bids = second_maximum_bids_iter

        last_surplus = [surplus[-1] for surplus in agents_overall_surplus]
        if debug:
            print(f"\teach agent's surplus: {last_surplus}")
            print(f"\tsums to {np.array(last_surplus).sum()}")
        
        # contexts log
        contexts_logs.extend([opp.context for opp in agents[0].logs])
        agents_bids_logs.extend( [ [agent.logs[i].bid for agent in agents]  for i in range(len(agent.logs)) ] )

        # Update agents
        # Clear running metrics
        # TODO: update in parallel using concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(auction.agents)) as executor:
            res = {executor.submit(agent_update, agent, i) : agent for agent in auction.agents}
        

        # OLD (SEQUENTIAL)
        # for agent_id, agent in enumerate(auction.agents):
        #     if(len(agent.logs)>0):
        #         if debug: print(f'\t agent update: {my_agents_names[agent_id]}')
        #         agent.update(iteration=i)
        #         # if i==num_iter-1:
        #         #     agents_last_avg_utilities[agent_id].append(agent.bidder.expected_utilities)
        #         agent.clear_utility()
        #         agent.clear_logs()

        # Log revenue
        auction_revenue.append(auction.revenue)
        auction.clear_revenue()

        # Update progress bar
        # pbar.update()
    
    # regret retrievement
    for agent_id, agent in enumerate(auction.agents):
        agents_regret_history[agent_id] = agent.bidder.regret
        agents_actionsrewards_history[agent_id] = agent.bidder.actions_rewards
        pass

    # Rescale metrics per auction round
    auction_revenue = np.array(auction_revenue, dtype=object) / rounds_per_iter
    social_welfare = np.array(social_welfare, dtype=object) / rounds_per_iter
    advertisers_surplus = np.array(advertisers_surplus, dtype=object) / rounds_per_iter

    ### SAVE UPDATE LOGS
    #make dir if not exists
    folder = folder_name if not None else ROOT_DIR / "src/results" / config_name / ts
    with open(folder / f"{num_run}_update_logs.txt", 'w') as f:
        for log in agents_update_logs:
            f.write(log + '\n')

    ### SAVE CONTEXTS LOGS
    contexts_logs = np.array(contexts_logs, dtype=object)
    np.save(folder / f"{num_run}_contexts_logs.npy", contexts_logs)

    ### SAVE BIDS LOGS
    agents_bids_logs = np.array(agents_bids_logs, dtype=object)
    np.save(folder / f"{num_run}_agents_bids_logs.npy", agents_bids_logs)
    with open(folder / f"{num_run}_agents_bids_logs.txt", 'w') as f:
        for log in agents_bids_logs:
            f.write(str(log) + '\n')



    ### SECONDARY OUTPUTS ###
    # secondary_outputs.append((agents_last_avg_utilities, [a.bidder.BIDS for a in auction.agents]))
    if results is not None:
        results[num_run] = (
            auction_revenue, social_welfare, advertisers_surplus, 
            agents_overall_surplus, agents_instant_surplus, 
            agents_regret_history, agents_actionsrewards_history
        )
    

    return auction_revenue, social_welfare, advertisers_surplus,\
            agents_overall_surplus, agents_instant_surplus,\
            agents_regret_history, agents_actionsrewards_history


def construct_graph(data, graph, xlabel, ylabel, insert_labels=False, fontsize=16):
    # data = np.array([x[index] for x in num_participants_2_metrics]).squeeze().transpose(1,0,2)

    y_err = []
    for i, agent in enumerate(data):
        y_err.append(agent.std(axis=0) / np.sqrt(num_runs))
        # data[i] = d.mean(axis=0)      # WHY TO DO THAT

    y = []
    for i, agent in enumerate(data):
        y.append(agent.mean(axis=0))

    for i, agent in enumerate(y):
        graph.plot(agent, label=my_agents_names[i])
        graph.fill_between(range(len(agent)), agent-y_err[i], agent+y_err[i], alpha=0.2)

    graph.set_xlabel(xlabel, fontsize=fontsize)
    graph.set_ylabel(ylabel, fontsize=fontsize)
    graph.set_xticks(list(range(0,num_iter,25)))
    graph.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    graph.axhline(0, color='black', lw=1, alpha=.7)

    if insert_labels:
        handles, labels = graph.get_legend_handles_labels()
        legend = graph.legend(reversed(handles),
                                reversed(labels),
                                loc='upper left',
                                bbox_to_anchor=(1.0, 1.0),
                                fontsize=fontsize)
    

def show_graph(runs_results, filename="noname", printFlag=False):
    plt.ioff()

    # # Create a new figure, plot into it, then close it so it never gets displayed
    # fig = plt.figure()
    # plt.plot([1,2,3])
    # plt.savefig('/tmp/test0.png')
    # plt.close(fig)

    # # Create a new figure, plot into it, then don't close it so it does get displayed
    # plt.figure()
    # plt.plot([1,3,2])
    # plt.savefig('/tmp/test1.png')

    # # Display all "open" (non-closed) figures
    # plt.show()

    fontsize = 16
    fig = plt.gcf()
    fig.set_size_inches(32,18)
    fig.sharey = 'all'
    gs = fig.add_gridspec(3, 2)


    graph_cumulative_surpluses = fig.add_subplot(gs[0:1, 0:2])
    graph_cumulative_regrets = fig.add_subplot(gs[1:2, 0:2])
    graph_instant_surpluses = fig.add_subplot(gs[2:3, 0:1])
    graph_regrets_hindsight = fig.add_subplot(gs[2:3, 1:2])

    graph_cumulative_surpluses.set_title(graph_title, fontsize=fontsize+4)

    # revenue, welfare, agent, agents_surplus = num_participants_2_metrics 

    #cumulative surpluses
    cumulative_surpluses = np.array([x[idx_cumulative_surpluses] for x in runs_results]).squeeze().transpose(1,0,2)
    instant_surpluses = np.array([x[idx_instant_surpluses] for x in runs_results]).squeeze().transpose(1,0,2)
    instant_regrets = np.array([x[idx_regrets] for x in runs_results]).squeeze().transpose(1,0,2)
    construct_graph(cumulative_surpluses, graph_cumulative_surpluses, '', 'Cumulative Surplus', insert_labels=True, fontsize=fontsize)
    construct_graph(instant_surpluses, graph_instant_surpluses, '', 'Instant Surplus', insert_labels=False, fontsize=fontsize)
    construct_graph(instant_regrets, graph_regrets_hindsight, '', 'Instant Regret in Hindsight', insert_labels=True, fontsize=fontsize)

    #cumulative regrets
    regrets_cumul = np.zeros_like(instant_regrets)
    for i in range(instant_regrets.shape[0]):
        for j in range(instant_regrets.shape[1]):
            regrets_cumul[i][j] = np.array([instant_regrets[i][j][:h+1].sum() for h in range(instant_regrets.shape[2])])
    construct_graph(regrets_cumul, graph_cumulative_regrets, '', 'Cumulative Regret in Hindsight', insert_labels=True, fontsize=fontsize)


    fig.tight_layout()

    # plt.show()
    ts = time.strftime("%Y%m%d-%H%M", time.localtime())
    plt.savefig(filename)
    if args.printall: print(f"Plot saved to {filename}")
    plt.close(fig)


############ MAIN ############
if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default="SP_BIGPR", help='Path to config file')
    parser.add_argument('--nprox', type=int, default=1, help='Number of processors to use')
    parser.add_argument('--printall', action='store_const', const=True, help='Whether to print results')
    parser.add_argument('--oneitem', action='store_const', const=True, help='Whether all agents should only have one item')
    parser.add_argument('--sameitem', action='store_const', const=True, help='Whether all agents should compete with the same item')
    parser.add_argument('--iter', type=int, default=-1, help='overwrite num_iter from config file (at least 1)')
    parser.add_argument('--runs', type=int, default=-1, help='overwrite num_runs from config file (at least 2)')
    parser.add_argument('--no-save-results', action='store_const', const=True, help='whether to save results in files or not (e.g. don\'t save if debug)')
    parser.add_argument('--no-save-data', action='store_const', const=True, help='whether to save data (e.g. don\'t save if limited space)')
    parser.add_argument('--use-server-data-folder', action='store_const', const=True, help='whether to save data on the data folder (for server)')

    args = parser.parse_args()

    args.printall = bool(args.printall)
    args.oneitem = bool(args.oneitem)
    args.sameitem = bool(args.sameitem)
    args.no_save_results = bool(args.no_save_results)
    args.no_save_data = bool(args.no_save_data)
    args.use_server_data_folder = bool(args.use_server_data_folder)

    # compute ts of the run
    # create folder for output files
    config_name = Path(args.config).stem
    file_prefix = config_name+"/"+ts
    if args.use_server_data_folder:
        folder_name = Path("/data/rtb/results") / file_prefix 
    else:
        folder_name = ROOT_DIR / "src" / "results" / file_prefix
    os.makedirs(folder_name, exist_ok=True)

    log_file = open(folder_name / "log_file.txt", 'w')

    if args.printall: 
        print("### 1. parsing arguments ###")
        print(f'\tUsing config file: <<{args.config}>>')
        print(f'\tUsing <<{args.nprox}>> processors')
        print(f'\tPrinting results flag: <<{args.printall}>>')
        print(f'\tOverwriting one item flag: <<{args.oneitem}>>, same item flag: <<{args.sameitem}>>')
        print(f'\tOverwriting num_iter: <<{args.iter if args.iter >= 1 else "UNCHANGED"}>>',
                            f', num_runs: <<{args.runs if args.runs >= 2 else "UNCHANGED"}>>')
        print(f'\tSaving results flag: <<{not args.no_save_results}>>, saving data flag: <<{not args.no_save_data}>>')
        print()
        print()
    
    #logging
    log_file.write("### 1. parsing arguments ###\n"+
                    f'\tUsing config file: <<{args.config}>>\n'+
                    f'\tUsing <<{args.nprox}>> processors\n'+
                    f'\tPrinting results flag: <<{args.printall}>>\n'+
                    f'\tOverwriting one item flag: <<{args.oneitem}>>, same item flag: <<{args.sameitem}>>\n'+
                    f'\tOverwriting num_iter: <<{args.iter if args.iter >= 1 else "UNCHANGED"}>>'+
                    f', num_runs: <<{args.runs if args.runs >= 2 else "UNCHANGED"}>>\n'+
                    f'\tSaving results flag: <<{not args.no_save_results}>>, saving data flag: <<{not args.no_save_data}>>\n'+
                    "\n\n"
                    )
    
    
    #
    # 2. config file
    #
    if args.printall: print("### 2. selecting config file ###")
    config_name = Path(args.config).stem
    config_file = ROOT_DIR / "config-mine" / (args.config + ".json")
    graph_title = config_file
    if args.printall: print(f'\tUsing config file: {args.config}')
    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("### 2. selecting config file ###\n"+
                    f'\tUsing config file: {args.config}\n'+
                    "\n\n"
                    )

    # 
    # 3. Parsing config file
    # 
    if args.printall: print("### 3. parsing config file ###")
    rng, config, agent_configs, agents2items, agents2item_values,\
    num_runs, max_slots, embedding_size, embedding_var,\
    obs_embedding_size = parse_config(config_file)

    if args.iter >= 1: config['num_iter'] = args.iter
    if args.runs >= 2: config['num_runs'] = args.runs
    num_iter = config['num_iter']
    num_runs = config['num_runs']

    if args.printall: print('--- Auction ---')
    if args.printall: print(config['allocation'])
    if args.printall: print()

    if args.printall: print('--- My Agents ---')
    my_agents_names = []
    i=0
    for agent in config['agents']:
        for copies in range(agent['num_copies']):
            i+=1
            # my_agents_names.append(f'{i}.{agent["bidder"]["type"]} ({agent["name"]})')
            my_agents_names.append(f'{i}. {agent["name"]}')
            # if args.printall: print(f'{i}) {agent["bidder"]["type"]}')
    if args.printall: print(my_agents_names)

    if args.printall: print()
    if args.printall: print('--- Runs Number ---')
    if args.printall: print(f"making {config['num_runs']} runs\n  for each, {config['num_iter']} iterations\n    for each, {config['rounds_per_iter']} episodes")
    if args.printall: print(f"\t -> total: {config['num_runs']*config['num_iter']*config['rounds_per_iter']}")

    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("### 3. parsing config file ###\n"+
                    '--- Auction ---\n'+
                    f'{config["allocation"]}' + "\n\n"+
                    '--- My Agents ---\n'+
                    f"{my_agents_names}\n\n"+
                    '--- Runs Number ---\n'+
                    f"making {config['num_runs']} runs\n  for each, {config['num_iter']} iterations\n    for each, {config['rounds_per_iter']} episodes\n"+
                    f"\t -> total: {config['num_runs']*config['num_iter']*config['rounds_per_iter']}\n"+
                    "\n\n"
                    )

    #
    # 4. overwriting products
    #
    if args.printall: print("### 4. overwriting products ###")
    ALL_AGENT_SAME_ITEM = args.sameitem
    REDUCE_TO_ONE_ITEM = args.oneitem
    if args.printall: print(f"overwrite products policy -> reduce to one prod: {REDUCE_TO_ONE_ITEM}, all agents same prod: {ALL_AGENT_SAME_ITEM}")
    agents_names = list(agents2items.keys())
    assert agents_names[0] == list(agents2item_values.keys())[0] 

    if ALL_AGENT_SAME_ITEM:     #assigns agent 1 items to all agents
        if args.printall: print("APPLYING: all agents same items")
        agents2items = { agent_name: agents2items[agents_names[0]] for agent_name in agents_names }
        agents2item_values = { agent_name: agents2item_values[agents_names[0]] for agent_name in agents_names }

    if REDUCE_TO_ONE_ITEM:      # only keeps first item for each agent
        if args.printall: print("APPLYING: reduce to one item per agent")
        agents2items = { agent_name: agents2items[agent_name][:1] for agent_name in agents_names }
        agents2item_values = { agent_name: agents2item_values[agent_name][:1] for agent_name in agents_names }

    # obj_embed, obj_value
    for agent_name in agents_names:
        if args.printall: print(agents2items[agent_name], " -> ", agents2item_values[agent_name])
    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("### 4. overwriting products ###\n"+
                    f"overwrite products policy -> reduce to one prod: {REDUCE_TO_ONE_ITEM}, all agents same prod: {ALL_AGENT_SAME_ITEM}\n"+
                    f"agents2items:\n{agents2items}\n\n"+
                    f"agents2item_values:\n{agents2item_values}\n\n"+
                    "\n\n"
                    )

    #
    # 5. running experiment
    #
    if args.printall: print("### 5. running experiment ###")

    import multiprocessing as mp
    import concurrent.futures
    secondary_outputs = []
    debug=False
    instantiate_agents_args = (rng, agent_configs, agents2item_values, agents2items)
    instantiate_auction_args = (config, max_slots, embedding_size, embedding_var, obs_embedding_size)

    arg_num_run = list(range(num_runs))
    # arg_pbars = {
    #     num_run: tqdm(total=num_iter, desc=f'{num_run+1}/{num_runs}', leave=True, colour="#0000AA") for num_run in arg_num_run
    # }

    runs_results = [None for _ in range(num_runs)]

    # # now using futures
    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.nprox) as executor:
    #     future_to_arg = {executor.submit(
    #                             run_repeated_auctions,
    #                                 num_run, num_runs,
    #                                 instantiate_agents_args, instantiate_auction_args,
    #                                 arg_pbars[num_run],
    #                                 runs_results, debug
    #                                 ): num_run
    #                                 for num_run in arg_num_run    }
    #     for i, future in enumerate(concurrent.futures.as_completed(future_to_arg)):
    #         runs_results[i] = future.result()
    #         # pbar.update()
    #         # You can collect the results if needed
    #         # results.append(result)

    # #threads using mp
    # threads = [mp.Process(target=run_repeated_auctions, args=(i, num_runs, instantiate_agents_args, instantiate_auction_args, arg_pbars[i], runs_results, debug)) for i in range(num_runs)]

    # for t in threads:
    #     t.start()
    
    # for t in threads:
    #     t.join()

    # threads using ray
    ray.init()
    processes = [run_repeated_auctions.remote(i, num_runs, instantiate_agents_args, instantiate_auction_args, runs_results, debug) for i in tqdm(range(num_runs))]

    runs_results = []
    for i in tqdm(range(num_runs)):
        runs_results.append(ray.get(processes[i]))

    # runs_results = ray.get(processes)

    runs_results = np.array(runs_results, dtype=object)

    if args.printall: print("RUN IS DONE")
    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("### 5. running experiment ###\n"+
                    "RUN IS DONE\n"+
                    "\n\n"
                    )

    #
    # 6. print surpluses
    #
    if args.printall: print("### 6. saving results ###")
    if args.printall: print(my_agents_names)

    #logging
    log_file.write(f"### 6. saving results ###\n{my_agents_names}\n")
    
    total_surpluses = [[] for _ in range(len(my_agents_names))]

    # np.set_printoptions(precision=2, floatmode='fixed', sign=' ')

    for h, run in enumerate(runs_results):
        # r, w, s, a_s, i_s = run
        a_s = run[idx_cumulative_surpluses]
        i_s = run[idx_instant_surpluses]
        cumulatives = [np.float32(s[-1]).round(2) for s in  a_s]
        actions = np.array(run[idx_actions_rewards])[:,:,:,0].mean(axis=2).mean(axis=1)
        # surpluses = np.array([np.array(surp).sum().round(2) for surp in i_s], dtype=object)
        for i in range(len(i_s)):
            total_surpluses[i].append(cumulatives[i])

        # print_surpluses = ' '.join('{:7.2f}'.format(x) for x in surpluses)
        print_cumulatives = ' '.join('{:7.2f}'.format(x) for x in cumulatives)
        print_actions = ' '.join('{:7.2f}'.format(x) for x in actions)
        if args.printall: print(f'Run {h+1:=2}/{num_runs} -> surpluses: {print_cumulatives}\tactions: {print_actions}')
        
        #logging
        log_file.write(f'Run {h+1:=2}/{num_runs} -> surpluses: {print_cumulatives}\tactions: {print_actions}\n')

    # overall
    total_surpluses = np.array( [np.array(x).mean() for x in total_surpluses] )
    print_overall = ' '.join('{:7.2f}'.format(np.array(x).mean()) for x in total_surpluses)
    if args.printall: print('\n     PER-RUN AVERAGE: ', '[' + (print_overall) + ']')
    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write(
                    f"total_surpluses:\n{total_surpluses}\n"+
                    "PER-RUN AVERAGE:\n"+
                    f'[{print_overall}]\n\n\n'
                    )

    #
    # 7. save results
    #
    if args.printall: print(f"### skip saving results? {args.no_save_results} ###")

    #logging
    log_file.write(f"### skip saving results? {args.no_save_results} ###\n")

    if not args.no_save_results:
        if args.printall: print("### 7. saving results ###")

        # # save results
        # ts = time.strftime("%Y%m%d-%H%M", time.localtime())

        # #create folder
        # file_prefix = config_name+"/"+ts
        # folder_name = ROOT_DIR / "src" / "results" / file_prefix
        # os.makedirs(folder_name, exist_ok=True)

        results_filename = folder_name / "results.txt"
        with open(results_filename, 'w') as f:
            f.write(f'config: {args.config}\n')
            f.write(f'num_runs: {num_runs}\n')
            f.write(f'num_iter: {num_iter}\n')
            f.write(f'rounds_per_iter: {config["rounds_per_iter"]}\n')
            f.write(f'num_agents: {len(my_agents_names)}\n')
            f.write(f'agents: {my_agents_names}\n')
            f.write(f'agents2items: {agents2items}\n')
            f.write(f'agents2item_values: {agents2item_values}\n')
            f.write(f'embedding_size: {embedding_size}\n')
            f.write(f'embedding_var: {embedding_var}\n')
            f.write(f'obs_embedding_size: {obs_embedding_size}\n')
            f.write(f'\n\n\n')
        if args.printall: print("results saved in ", results_filename)

        # logging
        log_file.write("### 7. saving results ###\n"+
                        f'results saved in {results_filename}\n'+
                        "\n\n"
                        )

        if not args.no_save_data:
            if args.use_server_data_folder:
                data_folder = Path("/data/rtb") / file_prefix
                data_folder.mkdir(parents=True, exist_ok=True)
                data_filename = data_folder / "data.npy"
            else:
                data_filename = folder_name / "data.npy"
            np.save(data_filename, runs_results)
            if args.printall: print("data saved in ", data_filename)

            # logging
            log_file.write(f"data saved in {data_filename}\n")


        #
        # 8. save plot
        #
        if args.printall: print("### 8. saving plot ###")

        plot_filename = folder_name / "plot.png"
        show_graph(runs_results, plot_filename, args.printall)

        # logging
        log_file.write("### 8. saving plot ###\n"+
                        f'plot saved in {plot_filename}\n'+
                        "\n\n"
                        )

    end_time = time.time()
    ts_end = time.strftime("%Y%m%d-%H%M", time.localtime())
    if args.printall: print(f"### START OF RUN ###\n\t{ts}")
    if args.printall: print(f"### END OF RUN ###\n\t{ts_end}")
    if args.printall: print(f'### TOTAL TIME ###\n\t{end_time-start_time} seconds')
    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write(f"### START OF RUN ###\n\t{ts}\n"+
                    f"### END OF RUN ###\n\t{ts_end}\n"+
                    f'### TOTAL TIME ###\n\t{end_time-start_time} seconds\n'+
                    "\n\n"
                    )
    
    log_file.close()
