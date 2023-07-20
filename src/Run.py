import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from main import parse_config, instantiate_agents, instantiate_auction, simulation_run
from tqdm import tqdm
import time
from utils import get_project_root
from pathlib import Path

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


def run_repeated_auctions(num_run, num_runs, results=None, debug=False):
    # Placeholders for output
    auction_revenue = []
    social_welfare = []
    advertisers_surplus = []
    
    # Instantiate Agent and Auction objects
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    
    agents_overall_surplus = [[] for _ in range(len(agents))]

    agents_instant_surplus = [[] for _ in range(len(agents))]

    agents_regret_history = [[] for _ in range(len(agents))] #TODO
    agents_actionsrewards_history = [[] for _ in range(len(agents))] #TODO

    ### SECONDARY OUTPUTS ###
    agents_last_avg_utilities = [[] for _ in range(len(agents))]

    # Instantiate Auction object
    auction, num_iter, rounds_per_iter, output_dir =\
        instantiate_auction(rng,
                            config,
                            agents2items,
                            agents2item_values,
                            agents,
                            max_slots,
                            embedding_size,
                            embedding_var,
                            obs_embedding_size)
    
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
                    if args.print: print('\t', agent.name, ': ', agent.bidder.BIDS)
                else:
                    if args.print: print('\t', agent.name, ': ', agent.bidder.textContinuous)

    if debug:
        for agent in auction.agents:
            if args.print: print(agent.name, ': ', agent.bidder.auction_type, end=' | ')

    # Run repeated auctions
    # This logic is encoded in the `simulation_run()` method in main.py
    # if args.print: print(num_run, ') ', end='')
    for i in tqdm(range(num_iter), desc=f'{num_run+1}/{num_runs}', leave=True):
        if debug:
            if args.print: print(f'Iteration {i+1} of {num_iter}')

        # Simulate impression opportunities
        for _ in range(rounds_per_iter):
            auction.simulate_opportunity()

        # GET ALL AGENTS BIDS -> calculate winning bids and give to bidders
        # winning_iter_bids = np.zeros(rounds_per_iter, dtype=np.float32)
        iter_bids = [[] for _ in range(len(agents))]
        for agent_id, agent in enumerate(auction.agents):
            # winning_iter_bids = np.maximum(    winning_iter_bids, 
                                            # np.array(list(opp.bid for opp in agent.logs), dtype=object)     )
            iter_bids[agent_id] = np.array(list(opp.bid for opp in agent.logs), dtype=object)
        
        combined_array = np.vstack(iter_bids)
        sorted_bids_iter = np.sort(combined_array, axis=0)
        maximum_bids_iter = sorted_bids_iter[-1]
        second_maximum_bids_iter = sorted_bids_iter[-2]

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
            if args.print: print(f"\teach agent's surplus: {last_surplus}")
            if args.print: print(f"\tsums to {np.array(last_surplus).sum()}")
        
        # Update agents
        # Clear running metrics
        for agent_id, agent in enumerate(auction.agents):
            if(len(agent.logs)>0):
                if debug:
                    if args.print: print(f'\t agent update: {my_agents_names[agent_id]}')
                agent.update(iteration=i)
                # if i==num_iter-1:
                #     agents_last_avg_utilities[agent_id].append(agent.bidder.expected_utilities)
                agent.clear_utility()
                agent.clear_logs()

        # Log revenue
        auction_revenue.append(auction.revenue)
        auction.clear_revenue()
    
    # regret retrievement
    for agent_id, agent in enumerate(auction.agents):
        agents_regret_history[agent_id] = agent.bidder.regret
        agents_actionsrewards_history[agent_id] = agent.bidder.actions_rewards
        pass

    # Rescale metrics per auction round
    auction_revenue = np.array(auction_revenue, dtype=object) / rounds_per_iter
    social_welfare = np.array(social_welfare, dtype=object) / rounds_per_iter
    advertisers_surplus = np.array(advertisers_surplus, dtype=object) / rounds_per_iter

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
    if args.print: print(f"Plot saved to {filename}")
    plt.close(fig)


############ MAIN ############
if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default="SP_BIGPR", help='Path to config file')
    parser.add_argument('-nprox', type=int, default=1, help='Number of processors to use')
    parser.add_argument('-print', type=bool, default=False, help='Whether to print results')
    parser.add_argument('-oneitem', type=bool, default=True, help='Whether all agents should only have one item')
    parser.add_argument('-sameitem', type=bool, default=True, help='Whether all agents should compete with the same item')
    parser.add_argument('-iter', type=int, default=-1, help='overwrite num_iter from config file')
    args = parser.parse_args()
    if args.print: print("### 1. importings all done ###")
    if args.print: print()
    if args.print: print()
    
    #
    # 2. config file
    #
    if args.print: print("### 2. selecting config file ###")
    config_name = Path(args.config).stem
    config_file = ROOT_DIR / "config-mine" / (args.config + ".json")
    graph_title = config_file
    if args.print: print(f'\tUsing config file: {args.config}')
    if args.print: print()
    if args.print: print()

    # 
    # 3. Parsing config file
    # 
    if args.print: print("### 3. parsing config file ###")
    rng, config, agent_configs, agents2items, agents2item_values,\
    num_runs, max_slots, embedding_size, embedding_var,\
    obs_embedding_size = parse_config(config_file)

    num_iter = config['num_iter'] if args.iter <= 0 else args.iter

    if args.print: print('--- Auction ---')
    if args.print: print(config['allocation'])
    if args.print: print()

    if args.print: print('--- My Agents ---')
    my_agents_names = []
    i=0
    for agent in config['agents']:
        for copies in range(agent['num_copies']):
            i+=1
            # my_agents_names.append(f'{i}.{agent["bidder"]["type"]} ({agent["name"]})')
            my_agents_names.append(f'{i}. {agent["name"]}')
            # if args.print: print(f'{i}) {agent["bidder"]["type"]}')
    if args.print: print(my_agents_names)

    if args.print: print()
    if args.print: print('--- Runs Number ---')
    if args.print: print(f"making {config['num_runs']} runs\n  for each, {config['num_iter']} iterations\n    for each, {config['rounds_per_iter']} episodes")
    if args.print: print(f"\t -> total: {config['num_runs']*config['num_iter']*config['rounds_per_iter']}")

    if args.print: print()
    if args.print: print()

    #
    # 4. overwriting products
    #
    if args.print: print("### 4. overwriting products ###")
    ALL_AGENT_SAME_PRODUCT = args.sameitem
    REDUCE_TO_ONE_PRODUCT = args.oneitem
    agents_names = list(agents2items.keys())
    assert agents_names[0] == list(agents2item_values.keys())[0] 
    obj_embed = [ agents2items[ agent_name ] [0] for agent_name in agents_names]
    obj_value = [ agents2item_values[ agent_name ] [0] for agent_name in agents_names]

    if ALL_AGENT_SAME_PRODUCT:
        obj_embed = [ obj_embed[0] ] * len(obj_embed)
        obj_value = [ obj_value[0] ] * len(obj_value)

    if REDUCE_TO_ONE_PRODUCT:
        for i,a in enumerate(agents_names):
            agents2items[a] = np.array([obj_embed[i]])
            agents2item_values[a] = np.array([obj_value[i]])

    # obj_embed, obj_value
    for agent_name in agents_names:
        if args.print: print(agents2items[agent_name], " -> ", agents2item_values[agent_name])
    if args.print: print()
    if args.print: print()

    #
    # 5. running experiment
    #
    if args.print: print("### 5. running experiment ###")

    from threading import Thread
    secondary_outputs = []
    debug=False

    runs_results = [None for _ in range(num_runs)]

    threads = [Thread(target=run_repeated_auctions, args=(i, num_runs, runs_results, debug)) for i in range(num_runs)]

    n_prox = args.nprox

    i=0
    j=0
    while i < num_runs:
        # if args.print: print(i,' &&& ',j)
        for j in range(n_prox):
            if i+j >= len(threads):
                break
            threads[i+j].start()
            
        for j in range(n_prox):
            if i+j >= len(threads):
                break
            threads[i+j].join()
        
        i+=n_prox

    if args.print: print("RUN IS DONE")
    if args.print: print()
    if args.print: print()

    #
    # 6. print surpluses
    #
    if args.print: print("### 6. saving results ###")
    if args.print: print(my_agents_names)
    total_surpluses = [[] for _ in range(len(my_agents_names))]

    # np.set_printoptions(precision=2, floatmode='fixed', sign=' ')

    for h, run in enumerate(runs_results):
        # r, w, s, a_s, i_s = run
        a_s = run[idx_cumulative_surpluses]
        i_s = run[idx_instant_surpluses]
        cumulatives = [np.float32(s[-1]).round(2) for s in  a_s]
        # surpluses = np.array([np.array(surp).sum().round(2) for surp in i_s], dtype=object)
        # for i in range(len(i_s)):
        #     total_surpluses[i].append(surpluses[i])

        # print_surpluses = ' '.join('{:7.2f}'.format(x) for x in surpluses)
        print_cumulatives = ' '.join('{:7.2f}'.format(x) for x in cumulatives)
        if args.print: print(f'Run {h+1:=2}/{num_runs} -> surpluses: {print_cumulatives}')

    # overall
    total_surpluses = np.array( [np.array(x).mean() for x in total_surpluses] )
    print_overall = ' '.join('{:7.2f}'.format(np.array(x).mean()) for x in total_surpluses)
    if args.print: print('\n     PER-RUN AVERAGE: ', '[' + (print_overall) + ']')
    if args.print: print()
    if args.print: print()


    #
    # 7. save results
    # 
    if args.print: print("### 7. saving results ###")

    # save results
    ts = time.strftime("%Y%m%d-%H%M", time.localtime())

    #create folder
    file_prefix = config_name+"/"+ts
    folder_name = ROOT_DIR / "src" / "results" / file_prefix
    os.makedirs(folder_name, exist_ok=True)

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

    if args.print: print("results saved in ", results_filename)

    data_filename = folder_name / "data.npy"
    np.save(data_filename, runs_results)

    if args.print: print("data saved in ", data_filename)


    #
    # 8. save plot
    #
    if args.print: print("### 8. saving plot ###")

    plot_filename = folder_name / "plot.png"
    show_graph(runs_results, plot_filename, args.print)
