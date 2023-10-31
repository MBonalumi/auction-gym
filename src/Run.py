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
import utils
from pathlib import Path
from shutil import copy2 as copyfile

import concurrent.futures

import joblib

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

# CONSTANTS
ROOT_DIR = utils.get_project_root()

# INDEXES of the return
idx_auction_rev = 0
idx_social_welfare = 1
idx_advertisers_surplus = 2
idx_cumulative_surpluses = 3
idx_instant_surpluses = 4
idx_regrets = 5
idx_actions_rewards = 6
idx_cv_regret = 7
idx_bids = 8
idx_contexts = 9
idx_ctrs = 10

# LOGS
agents_update_logs = []
agents_update_logs_npy = None
# contexts_logs = []
# agents_bids_logs = []
my_agents_names = []


@ray.remote
def run_repeated_auctions_remote(num_run, num_runs, instantiate_agents_args, instantiate_auction_args,
                            clairevoyant, clairevoyant_type, clear_results=False, results=None, debug=False):
    return run_repeated_auctions(num_run, num_runs, instantiate_agents_args, instantiate_auction_args,
                            clairevoyant, clairevoyant_type, clear_results, results, debug)

def run_repeated_auctions(num_run, num_runs, instantiate_agents_args, instantiate_auction_args,
                            clairevoyant, clairevoyant_type, clear_results=False, results=None, debug=False):
    
    ### 0. INITIALIZATIONS

    rng, agent_configs, agents2item_values, agents2items = instantiate_agents_args
    config, max_slots,embedding_size,embedding_var,obs_embedding_size = instantiate_auction_args

    # replace rng, to have different seeds for each run
    seed = config['random_seed'] + num_run
    print(f"Run #{num_run}\t seed: {seed}")
    del rng
    rng = np.random.default_rng(seed)
    np.random.seed(seed)


    ### 1. OUTPUTS DECLARATION

    auction_revenue = []
    social_welfare = []
    advertisers_surplus = []
    
    # Instantiate Agent and Auction objects
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    
    agents_overall_surplus = [[] for _ in range(len(agents))]

    agents_instant_surplus = [[] for _ in range(len(agents))]

    agents_regret_history = [[] for _ in range(len(agents))] 
    agents_actionsrewards_history = [[] for _ in range(len(agents))]

    clairevoyant_regret = []
    ctrs = []
    agents_bids = [[] for _ in range(len(agents))]

    # Instantiate Auction object
    auction, num_iter, rounds_per_iter, output_dir = \
        instantiate_auction(rng,
                            config,
                            agents2items, agents2item_values, agents,
                            max_slots,
                            embedding_size, embedding_var, obs_embedding_size)
    

    ### 2. INITIALIZE BIDDERS VARIABLES - FOR INFO ABOUT THE CONFIGURATION

    # give bidder info about the auction type (2nd price, 1st price, etc.) to compute 'regret in hindsight'
    from BidderBandits import BaseBidder, StaticBidder
    from BidderNovelty import NoveltyClairevoyant
    for id, agent in enumerate(auction.agents):
        if isinstance(agent.bidder, BaseBidder):
            agent.bidder.auction_type = config['allocation']
            agent.bidder.agent_id = id
            agent.bidder.num_iterations = num_iter
            # agent.bidder.total_num_auctions = num_iter * rounds_per_iter
            # agent.bidder.item_values = agent.item_values
            if not isinstance(agent.bidder, StaticBidder) and not isinstance(agent.bidder, NoveltyClairevoyant):
                # agent.bidder.clairevoyant = joblib.load(ROOT_DIR / "src" / "models" / "clairevoyant" / "20230912-1147.joblib")
                agent.bidder.clairevoyant = clairevoyant
                agent.bidder.clairevoyant_type = clairevoyant_type


            if num_run == 0: 
                if not agent.bidder.isContinuous:
                    print('\t', agent.name, f' ({agent.bidder.agent_id}): ', agent.bidder.BIDS)
                else:
                    print('\t', agent.name, ': ', agent.bidder.textContinuous)

    if debug:
        for agent in auction.agents:
            print(agent.name, ': ', agent.bidder.auction_type, end=' | ')


    ### 3. RUN THE REPEATED AUCTIONS

    # This logic is encoded in the `simulation_run()` method in main.py
    # print(num_run, ') ', end='')
    # from tqdm import tqdm
    # with tqdm(total=num_iter, desc=f'{num_run+1}/{num_runs}', leave=True) as pbar:
    for iteration in range(num_iter):
        if debug: print(f'Iteration {iteration+1} of {num_iter}')
        if iteration % int(num_iter/10) == 0: print(f'Run #{num_run}\t {iteration/num_iter*100:.2f}%')

        # Simulate impression opportunities
        opportunities_results = []  
        for _ in range(rounds_per_iter):
            opportunities_results.append( auction.simulate_opportunity() )
        
        # participating_agents_ids = np.array(np.array(opportunities_results)[:,0,:], dtype=np.int32)
        iter_bids = np.array(np.array(opportunities_results)[:,1,:], dtype=np.float32)
        
        # participating_agents_masks = [np.isin(participating_agents_ids, agent).any(axis=1) for agent in range(len(agents))]

        sorted_bids_iter = np.sort(iter_bids, axis=1)
        maximum_bids_iter = sorted_bids_iter[:,-1]
        second_maximum_bids_iter = sorted_bids_iter[:,-2]

        social_welfare.append(sum([agent.gross_utility for agent in auction.agents]))
        advertisers_surplus.append(sum([agent.net_utility for agent in auction.agents]))

        # give info about the bids to the bidders, to compute regret in hindsight
        # NOTE:  why not doing it here instead that in the bidders code?  wanted to keep the gym code as untouched as possible! 
        for agent_id, agent in enumerate(auction.agents):
            agent.bidder.winning_bids = maximum_bids_iter
            agent.bidder.second_winning_bids = second_maximum_bids_iter

        # Update agents
        # Clear running metrics
        
        for agent in auction.agents:
            if len(agent.logs) > 0:
                agent.update(iteration=iteration)
                agent.clear_logs()
                agent.clear_utility()

        if clear_results:
            for agent in auction.agents:
                del agent.bidder.regret[:]
                del agent.bidder.actions_rewards[:]
                del agent.bidder.surpluses[:]
                del agent.bidder.clairevoyant_regret[:]
        
        # Log revenue
        auction_revenue.append(auction.revenue)
        auction.clear_revenue()


    ### RETRIEVE DATA FROM BIDDERS
    
    for agent_id, agent in enumerate(auction.agents):
        agents_instant_surplus[agent_id] = agent.bidder.surpluses
        agents_overall_surplus[agent_id] = np.array(agents_instant_surplus[agent_id]).cumsum()
        agents_regret_history[agent_id] = agent.bidder.regret
        agents_actionsrewards_history[agent_id] = agent.bidder.actions_rewards
        if not isinstance(agent.bidder, StaticBidder):
            clairevoyant_regret.append(agent.bidder.clairevoyant_regret)
            ctrs.append(agent.bidder.ctrs)
        agents_bids[agent_id].append(agent.bidder.bids)
    contexts = auction.agents[-1].bidder.contexts

    # Rescale metrics per auction round
    auction_revenue = np.array(auction_revenue, dtype=object) / rounds_per_iter
    social_welfare = np.array(social_welfare, dtype=object) / rounds_per_iter
    advertisers_surplus = np.array(advertisers_surplus, dtype=object) / rounds_per_iter


    ### RETURN DATA

    if results is not None:
        results[num_run] = (
            auction_revenue, social_welfare, advertisers_surplus, 
            agents_overall_surplus, agents_instant_surplus, 
            agents_regret_history, agents_actionsrewards_history,
            clairevoyant_regret, agents_bids, contexts, ctrs
        )
    
    return auction_revenue, social_welfare, advertisers_surplus,\
            agents_overall_surplus, agents_instant_surplus,\
            agents_regret_history, agents_actionsrewards_history,\
            clairevoyant_regret, agents_bids, contexts, ctrs


def construct_graph(data, graph, xlabel, ylabel, names=my_agents_names, insert_labels=False, fontsize=16, moving_average=1):
    # data = np.array([x[index] for x in num_participants_2_metrics]).squeeze().transpose(1,0,2)

    y_err = []
    for i, agent in enumerate(data):
        y_err.append(agent.std(axis=0) / np.sqrt(num_runs))
        # data[i] = d.mean(axis=0)      # WHY TO DO THAT

    y = []
    for i, agent in enumerate(data):
        y.append(agent.mean(axis=0))

    for i, agent in enumerate(y):
        agent_ma = np.convolve(agent, np.ones(moving_average), 'valid') / moving_average if moving_average > 1 else agent
        y_err_ma = np.convolve(y_err[i], np.ones(moving_average), 'valid') / moving_average if moving_average > 1 else y_err[i]
        graph.plot(agent_ma, label=names[i])
        graph.fill_between(range(len(agent_ma)), agent_ma-y_err_ma, agent_ma+y_err_ma, alpha=0.2)

    graph.set_xlabel(xlabel, fontsize=fontsize)
    graph.set_ylabel(ylabel, fontsize=fontsize)
    data_amt = len(y[0])
    graph.set_xticks(list( range(0, data_amt, int(data_amt/20) ) ))
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
    fig.set_size_inches(40,22)
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

    print(f"cumulative_surpluses.shape: {cumulative_surpluses.shape}, instant_surpluses.shape: {instant_surpluses.shape}, instant_regrets.shape: {instant_regrets.shape}")
    construct_graph(cumulative_surpluses, graph_cumulative_surpluses,
                    '', 'Cumulative Surplus',
                    insert_labels=True, fontsize=fontsize)
    construct_graph(instant_surpluses, graph_instant_surpluses,
                    '', 'Instant Surplus',
                    insert_labels=False, fontsize=fontsize, moving_average=instant_surpluses.shape[2]//50)
    construct_graph(instant_regrets, graph_regrets_hindsight,
                    '', 'Instant Regret in Hindsight',
                    insert_labels=True, fontsize=fontsize, moving_average=instant_regrets.shape[2]//50)

    #cumulative regrets
    regrets_cumul = np.cumsum(instant_regrets, axis=2)
    construct_graph(regrets_cumul, graph_cumulative_regrets,
                    '', 'Cumulative Regret in Hindsight',
                    insert_labels=True, fontsize=fontsize)

    fig.tight_layout()

    ts = time.strftime("%Y%m%d-%H%M", time.localtime())
    #save in high resolution
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    if args.printall: print(f"Plot saved to {filename}")
    plt.close(fig)


############ MAIN ############
if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default="config-mine/SP_BIGPR", help='Path to config file')
    parser.add_argument('--nprox', type=int, default=1, help='Number of processors to use')
    parser.add_argument('--printall', action='store_const', const=True, help='Whether to print results')
    parser.add_argument('--oneitem', action='store_const', const=True, help='Whether all agents should only have one item')
    parser.add_argument('--sameitem', action='store_const', const=True, help='Whether all agents should compete with the same item')
    parser.add_argument('--iter', type=int, default=-1, help='overwrite num_iter from config file (at least 1)')
    parser.add_argument('--runs', type=int, default=-1, help='overwrite num_runs from config file (at least 2)')
    parser.add_argument('--no-save-results', action='store_const', const=True, help='whether to save results in files or not (e.g. don\'t save if debug)')
    parser.add_argument('--save-data', action='store_const', const=True, help='whether to save data (e.g. don\'t save if limited space)')
    parser.add_argument('--use-server-data-folder', action='store_const', const=True, help='whether to save data on the data folder (for server)')
    parser.add_argument('--no-plot', action='store_const', const=True, help='tell the script not to draw the plot of the results')
    parser.add_argument('--setting', type=str, default="default", help='setting name of the experiment, helps choosing clairevoyant model\nnow supported:\n\tfive_gaussians_staticbidders\n\tone_gaussians_staticbidders\n\tnoncontextual_bestbid')
    parser.add_argument('--clear-results', action='store_const', const=True, help='clears results folder before running the experiment')
    parser.add_argument('--serialize-runs', action='store_const', const=True, help='if added, serialize the runs results instead of running in parallel with ray')
    parser.add_argument('--discretize-ctxt', action='store_const', const=True, default=False, help='whether to discretize the context generated from the gym')
    parser.add_argument('--loosen-ctr', action='store_const', const=True, default=False, help='whether to loosen the CTRs generated from the gym (gym by default reduces CTRs distribution)')

    args = parser.parse_args()

    args.printall = bool(args.printall)
    args.oneitem = bool(args.oneitem)
    args.sameitem = bool(args.sameitem)
    args.no_save_results = bool(args.no_save_results)
    args.save_data = bool(args.save_data)
    args.use_server_data_folder = bool(args.use_server_data_folder)
    args.no_plot = bool(args.no_plot)
    args.clear_results = bool(args.clear_results)
    args.serialize_runs = bool(args.serialize_runs)
    args.discretize_ctxt = bool(args.discretize_ctxt)
    args.loosen_ctr = bool(args.loosen_ctr)
    
    discrete_cv = True

    if not discrete_cv:
        clairevoyant_type = "model"

        setting_to_clairevoyant = {
            "default" : "noncontextual_bestbid_5arms.joblib",

            "five_gaussians_staticbidders": "five_gaussians_staticbidders.joblib",              #TODO: maybe
            "one_gaussian_staticbidders": "one_gaussian_staticbidders.joblib",                  #TODO: maybe
            
            "noncontextual_bestbid": "noncontextual_bestbid_5arms.joblib",                      #done!
            "noncontextual_bestbid_withnoise": "noncontextual_bestbid_5arms_wNoise.joblib",     #done!
            
            "contextual_bestbid": "contextual_bestbid_5arms.joblib",                            #done!
            "contextual_bestbid_withnoise": "contextual_bestbid_5arms_wNoise.joblib",           #done!
            "contextual_bestbid_withnoise_scaledupctr": "contextual_bestbid_wNoise_scaledUpCTR.joblib",     #done!
        }
        clairevoyant_filename = ROOT_DIR / "src" / "clairevoyants" / setting_to_clairevoyant[args.setting]
        clairevoyant = joblib.load(clairevoyant_filename)
    
    elif discrete_cv:
        clairevoyant_type = "bestbid"

        setting_to_clairevoyant = {
            "default" : "ctxt_clairevoyant.npy",
            
            "noncontextual": "nonctxt_clairevoyant.npy",
            "contextual": "ctxt_clairevoyant.npy",
        }
        clairevoyant_filename = ROOT_DIR / "src" / "discr_clairevoyants" / setting_to_clairevoyant[args.setting]
        clairevoyant = np.load(clairevoyant_filename, allow_pickle=True)
        
        utils.create_config_file(discretized=args.discretize_ctxt, ctr_loosen=args.loosen_ctr)

        print(f"context will be discretized   -->   {args.discretize_ctxt}")
        print(f"CTR will be loosened   -->   {args.loosen_ctr}")

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
        print(f'\tSaving results flag: <<{not args.no_save_results}>>, saving data flag: <<{not args.save_data}>>')
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
                    f'\tSaving results flag: <<{not args.no_save_results}>>, saving data flag: <<{not args.save_data}>>\n'+
                    "\n\n"
                    )
    
    
    #
    # 2. config file
    #
    if args.printall: print("### 2. selecting config file ###")
    config_name = Path(args.config).stem
    config_file = ROOT_DIR / (args.config + ".json")
    #copy config file to results folder, as config.json
    #use os.copyfile instead of shutil.copyfile to avoid error if file already exists
    copyfile(config_file, folder_name / "config.txt")
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
    if args.runs >= 1: config['num_runs'] = args.runs
    num_iter = config['num_iter']
    num_runs = config['num_runs']

    if args.printall: print('--- Auction ---')
    if args.printall: print(config['allocation'])
    if args.printall: print()

    if args.printall: print('--- My Agents ---')
    # my_agents_names = []
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
        agents2items = { agent_name: agents2items[agents_names[-1]] for agent_name in agents_names }
        agents2item_values = { agent_name: agents2item_values[agents_names[-1]] for agent_name in agents_names }

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
                    f"agents2items:\n")
    for agent_name in agents_names:
        log_file.write( f"{agent_name}\n")
        for i in range(len(agents2items[agent_name])):
            log_file.write( f"\t{agents2items[agent_name][i]} -> {agents2item_values[agent_name][i]}\n")
    log_file.write("\n\n")

    #
    # 5. running experiment
    #
    if args.printall: print("### 5. running experiment ###")

    # import multiprocessing as mp
    # import concurrent.futures
    secondary_outputs = []
    debug=False
    instantiate_agents_args = (rng, agent_configs, agents2item_values, agents2items)
    instantiate_auction_args = (config, max_slots, embedding_size, embedding_var, obs_embedding_size)

    arg_num_run = list(range(num_runs))
    # arg_pbars = {
    #     num_run: tqdm(total=num_iter, desc=f'{num_run+1}/{num_runs}', leave=True, colour="#0000AA") for num_run in arg_num_run
    # }

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

    runs_results = []

    if not args.serialize_runs:
        ray.init()
        with tqdm(total=num_runs, desc=f'runs', leave=True, colour="#0000AA") as pbar:
            i=0
            while i < num_runs:
                processes = []

                prox_left = args.nprox if i+args.nprox <= num_runs else num_runs-i

                for j in range(prox_left):
                    if i+j > num_runs:
                        break
                    processes.append( 
                        run_repeated_auctions_remote.remote(i+j, num_runs, instantiate_agents_args, instantiate_auction_args,
                                                        clairevoyant=clairevoyant, clairevoyant_type=clairevoyant_type,
                                                        clear_results=args.clear_results, results=None, debug=debug)
                        )
                    
                
                for p in processes:
                    runs_results.append(ray.get(p))
                    pbar.update()
                    ray.cancel(p)

                i += prox_left
    
    elif args.serialize_runs:
        for i in tqdm(range(num_runs)):
            runs_results.append(
                run_repeated_auctions(i, num_runs, instantiate_agents_args, instantiate_auction_args,
                                        clairevoyant=clairevoyant, clairevoyant_type=clairevoyant_type,
                                        clear_results=args.clear_results, results=None, debug=debug)
                )
    
    runs_results = np.array(runs_results, dtype=object)

    if args.printall: print("RUN IS DONE")
    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("### 5. running experiment ###\n"+
                    "RUN IS DONE\n"+
                    "\n\n"
                    )


    #### IF args.clear_results, print nothing and exit
    if args.clear_results:
        print("now exiting...")
        exit(0)


    #
    # 6. print surpluses
    #
    if args.printall: print("### 6. saving results ###")
    if args.printall: print(my_agents_names)

    #logging
    log_file.write(f"### 6. saving results ###\n{my_agents_names}\n")
    

    # REWARDS
    total_surpluses = [[] for _ in range(len(my_agents_names))]

    # np.set_printoptions(precision=2, floatmode='fixed', sign=' ')

    for h, run in enumerate(runs_results):
        # r, w, s, a_s, i_s = run
        a_s = run[idx_cumulative_surpluses]
        i_s = run[idx_instant_surpluses]
        cumulatives = [np.float32(s[-1]).round(2) for s in  a_s]
        optimal_actions = np.array(run[idx_actions_rewards])[:,:,:,0].mean(axis=2).mean(axis=1)
        # surpluses = np.array([np.array(surp).sum().round(2) for surp in i_s], dtype=object)
        for i in range(len(i_s)):
            total_surpluses[i].append(cumulatives[i])

        # print_surpluses = ' '.join('{:7.2f}'.format(x) for x in surpluses)
        print_cumulatives = ' '.join('{:7.2f}'.format(x) for x in cumulatives)
        print_actions = ' '.join('{:7.2f}'.format(x) for x in optimal_actions)
        if args.printall: print(f'Run {h+1:=2}/{num_runs} -> surpluses: {print_cumulatives}\toptimal actions: {print_actions}')
        
        #logging
        log_file.write(f'Run {h+1:=2}/{num_runs} -> surpluses: {print_cumulatives}\toptimal actions: {print_actions}\n')

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

    # REGRETS
    if args.printall: print(my_agents_names)

    total_regrets = [[] for _ in range(len(my_agents_names))]

    for h, run in enumerate(runs_results):
        reg = run[idx_regrets]
        # print(reg[0])
        # print(reg[1])
        reg = np.array([np.array(r).sum() for r in reg])
        for i in range(len(reg)):
            total_regrets[i].append(reg[i])
        print_regrets = ' '.join('\t{:10.2f}'.format(x) for x in reg)
        if args.printall:print(f'Run {h+1:=2}/{num_runs} -> regrets: {print_regrets}')

        #logging
        log_file.write(f'Run {h+1:=2}/{num_runs} -> regrets: {print_regrets}\n')
    
    # overall
    avg_regrets = np.array([np.array(r).mean() for r in total_regrets])
    print_regrets = ' '.join('\t{:10.2f}'.format(x) for x in avg_regrets)
    # if args.printall: print('\n     PER-RUN AVERAGE: ', '[', reg, ']')
    if args.printall: print('\n     PER-RUN AVERAGE: ', '[' + (print_regrets) + ']')
    if args.printall: print()
    if args.printall: print()
    
    #logging
    log_file.write(
                    "PER-RUN AVERAGE:\n"+
                    f'[{print_regrets}]\n\n\n'
                    )

    # AVERAGE ACTION PLAYED
    if args.printall: print("AVERAGE ACTIONS PLAYED:")
    if args.printall: print(my_agents_names)
    
    runs_agents_bids = []

    for h, run in enumerate(runs_results):
        run_agents_bids = run[idx_bids]
        runs_agents_bids.append(run_agents_bids)

    if args.printall: 
        for h, run in enumerate(runs_agents_bids):
            print(f'Run {h+1:=2}/{num_runs} -> average actions:', end=' ')
            for agent_bids in run:
                print(f'{np.mean(agent_bids):.4f}', end=' ')
            print()
        print()

    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("AVERAGE ACTIONS PLAYED:\n"+
                    f"{my_agents_names}\n")
    for h, run in enumerate(runs_agents_bids):
        log_file.write(f'Run {h+1:=2}/{num_runs} -> average actions: ')
        for agent_bids in run:
            log_file.write(f'{np.mean(agent_bids):.4f} ')
        log_file.write('\n')


    # EXPLORATION OF SOME AUCTIONS
    if args.printall:
        SMALL_CONTEXTS = True
        rng0 = np.random.default_rng(42)

        # explore auction
        # intereseting data is: context -> bids -> winner -> regrets

        contexts = np.array([run[idx_contexts] for run in runs_results])    # runs, auctions   ->   (3, 10000)
        if SMALL_CONTEXTS:
            contexts = contexts[:,:,0]

        bids = np.array([run[idx_bids] for run in runs_results]).squeeze()
        bids = bids.transpose(1,0,2)            # users, runs, auctions   ->   (4, 3, 10000)

        regrets = np.array([run[idx_regrets] for run in runs_results])
        regrets = regrets.transpose(1,0,2)      # users, runs, auctions   ->   (4, 3, 10000)

        surpluses = np.array([run[idx_instant_surpluses] for run in runs_results])
        surpluses = surpluses.transpose(1,0,2)  # users, runs, auctions   ->   (4, 3, 10000)

        ctrs = np.array([run[idx_ctrs] for run in runs_results])
        ctrs = ctrs.transpose(1,0,2)            # users, runs, auctions   ->   (4, 3, 10000)

        exploring_run = 0
        exploring_auctions = rng0.choice( np.arange(len(contexts[exploring_run])), size=20, replace=False )

        to_print = ""

        for auct in exploring_auctions:
            to_print += f'auct:{auct:5}    ctxt={contexts[exploring_run,auct]:5.2f}    bids={bids[:,exploring_run,auct].round(2)}    winner={np.argmax(bids[:,exploring_run,auct])}    ctrs={ctrs[:,exploring_run,auct].round(3)}    reward={surpluses[:,exploring_run,auct].round(2)}    regret={regrets[:,exploring_run,auct].round(2)}\n'

        print(  f'run {exploring_run} \n'  )
        print(  to_print  )

    if args.printall: print()
    if args.printall: print()

    #logging
    log_file.write("EXPLORING SOME AUCTIONS:\n")
    log_file.write(f"exploring_run: {exploring_run}\n")
    log_file.write(f"{to_print}\n\n")


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

        if args.save_data:
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

        if args.printall: print()
        if args.printall: print()


        #
        # 8. save plot
        #
        if not args.no_plot:
            if args.printall: print("### 8. saving plot ###")

            plot_filename = folder_name / "1.plot.png"
            show_graph(runs_results, plot_filename, args.printall)

            # logging
            log_file.write("### 8. saving plot ###\n"+
                            f'plot saved in {plot_filename}\n'+
                            "\n\n"
                            )
            
            if args.printall: print()
            if args.printall: print()
        
        #
        # 9. plot my clairevoyant's regret
        #

        # take my agents names except those who contain the word "static", case insensitive
        if not args.no_plot:
            import re
            rule = re.compile(r'static', re.IGNORECASE)
            my_agents_names_no_static = [a for a in my_agents_names if not rule.search(a)]

            if args.printall: print("### 9. saving clairevoyant's regret plot ###")


            # 
            # 9.1 Instant Regret
            #

            regret_filename = folder_name / f"4.regret__{args.setting}__instant_aggregate.png"
            plt.ioff()

            fig, ax = plt.subplots(1,1, sharey='row', figsize=(20,10))

            ax.set_title(f"{args.config} Instantaneous Regret -vs- {args.setting} clairevoyant")

            clairevoyant_regret = np.array([r[idx_cv_regret] for r in runs_results]).transpose(1,0,2)
            data_amt = clairevoyant_regret.shape[2]
            # construct_graph(clairevoyant_regret, ax, 'iters', 'instant regret', 
            #                 names=my_agents_names_no_static, 
            #                 insert_labels=False, fontsize=16, moving_average=1)
            
            construct_graph(clairevoyant_regret, ax, 'iters', 'instant regret', 
                            names=[n+'_moving-avg'for n in my_agents_names_no_static], 
                            insert_labels=False, fontsize=16, moving_average=data_amt//100)

            plt.legend()
            plt.savefig(regret_filename, bbox_inches='tight', dpi=300)
            if args.printall: print(f"Plot saved to {regret_filename}")
            plt.close()


            # 
            # 9.2 Cumulative Regret
            #

            cumul_regret_filename = folder_name / f"5.regret__{args.setting}__cumulative_aggregate.png"
            plt.ioff()

            fig, ax = plt.subplots(1,1, sharey='row', figsize=(20,10))

            ax.set_title(f"{args.config} Cumulative Regret -vs- {args.setting} clairevoyant")
            ax.axline((0, 0), slope=1., color='grey', linestyle='--', linewidth=1)

            clairevoyant_regret_cumulative = clairevoyant_regret.cumsum(axis=2) 
            construct_graph(clairevoyant_regret_cumulative, ax, 'iters', 'cumulative regret', 
                            names=my_agents_names_no_static, 
                            insert_labels=False, fontsize=16, moving_average=1)
            # construct_graph(clairevoyant_regret_cumulative, ax, 'iters', 'cumulative regret', 
            #                 names=[n+'_moving-avg'for n in my_agents_names_no_static], 
            #                 insert_labels=False, fontsize=16, moving_average=moving_average)

            plt.legend()
            plt.savefig(cumul_regret_filename, bbox_inches='tight', dpi=300)
            if args.printall: print(f"Plot saved to {cumul_regret_filename}")
            plt.close()

            # logging
            log_file.write("### 9. saving clairevoyant's regret plot ###\n"+
                            f'plot saved in {cumul_regret_filename}\n'+
                            "\n\n"
                            )
            
            if args.printall: print()
            if args.printall: print()

            #
            # 9.3 Instant Regret for each context
            #
            SMALL_CONTEXTS_SETTING = True
            if args.discretize_ctxt:
                contexts = np.array([r[idx_contexts] for r in runs_results])
                if SMALL_CONTEXTS_SETTING:
                    contexts = contexts[:,:,0]
                instant_regrets = np.array([x[idx_regrets] for x in runs_results]).squeeze().transpose(1,0,2)
                mask = np.where(np.isin(my_agents_names, my_agents_names_no_static))
                instant_regrets = instant_regrets[mask].squeeze()
                
                contexts_vals = list(set(contexts.flatten()))

                contexts_vals_masks = [[None for _ in range(len(contexts_vals))] for _ in range(num_runs)]

                for r in range(num_runs):
                    for i,c in enumerate(contexts_vals):
                        contexts_vals_masks[r][i] = (contexts[r] == c)

                contexts_vals_masks = np.array(contexts_vals_masks).transpose(1,0,2)

                ir_contexts = [None for _ in range(len(contexts_vals))]

                for c in range(len(contexts_vals)):
                    ir_temp = [instant_regrets[r][contexts_vals_masks[c][r]] for r in range(num_runs)]
                    min_length = min([len(x) for x in ir_temp])
                    ir_temp = np.array([x[:min_length] for x in ir_temp])
                    ir_contexts[c] = ir_temp

                instant_regret_per_ctxt_filename = folder_name / "2.regret_GOD_instant_byContext.png"
                plt.ioff()
                fig, ax = plt.subplots(3,1, sharey='all', sharex='all', figsize=(20,18))
                ax[0].set_title("Instant Regret -vs- instant GODLY clairevoyant", fontsize=20)

                for i in range(len(contexts_vals)):
                    ir_temp = ir_contexts[i]
                    construct_graph(np.expand_dims(ir_temp, axis=0), ax[i], 'iters', f'context   {contexts_vals[i]:.2f}', 
                                names=my_agents_names_no_static, 
                                insert_labels=False, fontsize=16, moving_average=100)
                    
                plt.legend()
                plt.savefig(instant_regret_per_ctxt_filename, bbox_inches='tight', dpi=300)
                if args.printall: print(f"Plot saved to {instant_regret_per_ctxt_filename}")
                plt.close()


            #
            # 9.4 Cumulative Regret for each context
            #
                cr_contexts = [c.cumsum(axis=1) for c in ir_contexts]
                
                cumul_regret_per_ctxt_filename = folder_name / "3.regret_GOD_cumulative_byContext.png"
                plt.ioff()
                fig, ax = plt.subplots(3,1, sharey='all', sharex='all', figsize=(20,18))
                ax[0].set_title("Cumulative Regret -vs- cumulative GODLY clairevoyant", fontsize=20)

                for i in range(len(contexts_vals)):
                    cr_temp = cr_contexts[i]
                    ax[i].axline((0, 0), slope=1., color='grey', linestyle='--', linewidth=1)
                    construct_graph(np.expand_dims(cr_temp, axis=0), ax[i], 'iters', f'context   {contexts_vals[i]:.2f}', 
                                names=my_agents_names_no_static, 
                                insert_labels=False, fontsize=16, moving_average=100)
                    
                plt.legend()
                plt.savefig(cumul_regret_per_ctxt_filename, bbox_inches='tight', dpi=500)
                if args.printall: print(f"Plot saved to {cumul_regret_per_ctxt_filename}")
                plt.close()


            #
            # 9.5 Instant Regret for each context, wrt CUSTOM clairevoyant
            #
                cv_instant_regrets = np.array([r[idx_cv_regret] for r in runs_results]).transpose(1,0,2).squeeze()
                cv_ir_contexts = [None for _ in range(len(contexts_vals))]

                for c in range(len(contexts_vals)):
                    cv_ir_temp = [cv_instant_regrets[r][contexts_vals_masks[c][r]] for r in range(num_runs)]
                    min_length = min([len(x) for x in cv_ir_temp])
                    cv_ir_temp = np.array([x[:min_length] for x in cv_ir_temp])
                    cv_ir_contexts[c] = cv_ir_temp

                cv_instant_regret_per_ctxt_filename = folder_name / f"6.regret__{args.setting}__instant_byContext.png"
                
                plt.ioff()
                fig, ax = plt.subplots(3,1, sharey='all', sharex='all', figsize=(20,18))
                ax[0].set_title(f"Instant Regret -vs- clairevoyant {args.setting}", fontsize=20)

                for i in range(len(contexts_vals)):
                    cv_ir_temp = cv_ir_contexts[i]
                    construct_graph(np.expand_dims(cv_ir_temp, axis=0), ax[i], 'iters', f'context   {contexts_vals[i]:.2f}', 
                                names=my_agents_names_no_static, 
                                insert_labels=False, fontsize=16, moving_average=100)
                    
                plt.legend()
                plt.savefig(cv_instant_regret_per_ctxt_filename, bbox_inches='tight', dpi=300)
                if args.printall: print(f"Plot saved to {cv_instant_regret_per_ctxt_filename}")
                plt.close()


            #
            # 9.6 Cumulative Regret for each context, wrt CUSTOM clairevoyant
            #
                cv_cr_contexts = [c.cumsum(axis=1) for c in cv_ir_contexts]
                
                cv_cumul_regret_per_ctxt_filename = folder_name / f"7.regret__{args.setting}__cumulative_byContext.png"
                plt.ioff()
                fig, ax = plt.subplots(3,1, sharey='all', sharex='all', figsize=(20,18))
                ax[0].set_title(f"Cumulative Regret -vs- clairevoyant {args.setting}", fontsize=20)

                for i in range(len(contexts_vals)):
                    cv_cr_temp = cv_cr_contexts[i]
                    ax[i].axline((0, 0), slope=1., color='grey', linestyle='--', linewidth=1)
                    construct_graph(np.expand_dims(cv_cr_temp, axis=0), ax[i], 'iters', f'context   {contexts_vals[i]:.2f}', 
                                names=my_agents_names_no_static, 
                                insert_labels=False, fontsize=16, moving_average=100)
                    
                plt.legend()
                plt.savefig(cv_cumul_regret_per_ctxt_filename, bbox_inches='tight', dpi=500)
                if args.printall: print(f"Plot saved to {cv_cumul_regret_per_ctxt_filename}")
                plt.close()

    #
    # ENDING
    #
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
