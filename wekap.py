"""
Plot the fluxes and rates from direct.h5 files.

list(h5):
['arrivals', 'avg_color_probs', 'avg_conditional_fluxes', 'avg_rates', 
'avg_state_probs', 'avg_total_fluxes', 'color_prob_evolution', 'conditional_arrivals',
'conditional_flux_evolution', 'conditional_fluxes', 'duration_count', 'durations', 
'rate_evolution', 'state_labels', 'state_pop_evolution', 'target_flux_evolution', 'total_fluxes']

  /target_flux_evolution [window,state]
    Total flux into a given macro state based on
    windows of iterations of varying width, as in /rate_evolution.
  /conditional_flux_evolution [window,state,state]
    State-to-state fluxes based on windows of
    varying width, as in /rate_evolution.

The structure of these datasets is as follows:
  iter_start
    (Integer) Iteration at which the averaging window begins (inclusive).
  iter_stop
    (Integer) Iteration at which the averaging window ends (exclusive).
  expected
    (Floating-point) Expected (mean) value of the observable as evaluated within
    this window, in units of inverse tau.
  ci_lbound
    (Floating-point) Lower bound of the confidence interval of the observable
    within this window, in units of inverse tau.
  ci_ubound
    (Floating-point) Upper bound of the confidence interval of the observable
    within this window, in units of inverse tau.
  stderr
    (Floating-point) The standard error of the mean of the observable
    within this window, in units of inverse tau.
  corr_len
    (Integer) Correlation length of the observable within this window, in units
    of tau.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

#plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")
#plt.style.use("/Users/darian/github/wedap/styles/poster.mplstyle")

# TODO: function to make 4 panel plot
    # Plot of P_A, P_B, rate_AB, rate_BA, all as function of WE iteration
# TODO: allow a list of input files for multi kinetics runs with bayesian bootstrapping
# TODO: transition this to CLI program like mdap and wedap (eventually maybe make mkap)
class Kinetics:

    def __init__(self, direct="direct.h5", assign=None, statepop="direct", tau=100e-12, state=1, 
                 label=None, units="rates", ax=None, savefig=None, color=None, 
                 cumulative_avg=True, *args, **kwargs):
        """
        Parameters
        ----------
        direct : str
            Name of output direct.h5 file from WESTPA w_direct or w_ipa.
        assign : str
            Default None (and will search for a file names `assign.h5` in the same dir).
            Otherwise can specify a specific assign.h5 file. Needed for labeled population data.
        tau : float
            The resampling interval of the WE simualtion.
            This should be in seconds, default 100ps = 100 * 10^-12 (s).
        state : int
            State for flux calculations (flux into `state`), 0 = A and 1 = B.
        label : str
            Data label.
        statepop : str
            'direct' for state_population_evolution from direct.h5 or
            'assign' for labeled_populations from assign.h5.
        units : str
            Can be `rates` (default) or `mfpts`.
        ax : mpl axes object
        savefig : str
            Path to optionally save the figure.
        color : str
            Color of the line plot. Default None for mpl tab10 colors.
        cumulative_avg : bool
            Set to True (default) when kinetics were calculated with cumulative averaging.
        """
        # read in direct.h5 file
        self.direct_h5 = h5py.File(direct, "r")
        if assign is None:
            # temp solution for getting assign.h5, eventually can just get rid of it
            # since I don't think the color/labeled population is as useful
            self.assign_h5 = h5py.File(direct[:-9] + "assign.h5", "r")
        else:
             self.assign_h5 = assign

        self.tau = tau
        self.state = state
        self.label = label
        self.units = units
        self.statepop = statepop
        self.color = color

        if self.statepop == "direct":
            # divide k_AB by P_A for equilibrium rate correction (AB and BA steady states)
            self.state_pops = np.array(self.direct_h5["state_pop_evolution"])
            # state A = label 0, state B = label 1
            self.state_pop_a = np.array([expected[2] for expected in self.state_pops[:,0]])
            self.state_pop_b = np.array([expected[2] for expected in self.state_pops[:,1]])
        elif self.statepop == "assign":
            # divide k_AB by P_A for equilibrium rate correction (AB and BA steady states)
            self.state_pops = np.array(self.assign_h5["labeled_populations"])

            # when using cumulative averaging
            if cumulative_avg:
                # Replace 0 with the index of your source state here, 
                # the order you defined states in west.cfg.
                state_pop = self.assign_h5['labeled_populations'][:,0]
                temp = np.sum(state_pop, axis=1)
                #state_pop_cum_avg = np.cumsum(temp) / np.arange(1, len(temp)+1)
                self.state_pop_a = np.cumsum(temp) / np.arange(1, len(temp)+1)

                # state b
                state_pop = self.assign_h5['labeled_populations'][:,0]
                temp = np.sum(state_pop, axis=1)
                #state_pop_cum_avg = np.cumsum(temp) / np.arange(1, len(temp)+1)
                self.state_pop_a = np.cumsum(temp) / np.arange(1, len(temp)+1)
            # or if you're using e.g. instantaneous fluxes, can grab the state pops directly
            else:
                # state A = label 0, state B = label 1
                #self.state_pop_a = np.array([expected[0] for expected in self.state_pops[:,0]])
                #self.state_pop_a = np.sum(self.state_pops[:,0], axis=1)
                #print(state_pop_a)
                #np.sum(self.state_pop_a)
                self.state_pop_a = np.sum(self.state_pops[:,0], axis=1)
                self.state_pop_b = np.sum(self.state_pops[:,1], axis=1)

        # create new fig or plot onto existing
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = plt.gcf()

        self.savefig = savefig

    def extract_rate(self):
        """
        Get the raw rate array from one direct.h5 file.
        
        Returns
        -------
        rate_ab, ci_lb_ab, ci_ub_ab
        """

        # flux evolution dataset from cumulative evolution mode:
        # When calculating time evolution of rate estimates, 
        # ``cumulative`` evaluates rates over windows starting with --start-iter and 
        # getting progressively wider to --stop-iter by steps of --step-iter.
        fluxes = np.array(self.direct_h5["target_flux_evolution"])

        # conditional fluxes are macrostate to macrostate
        # 2 dimensions: [(0 -> 0, 0 -> 1), 
        #                (1 -> 0, 1 -> 1)] 
        # I want 0 -> 1
        #fluxes = np.array(h5["conditional_flux_evolution"])[:,:,1]

        # third column (expected) of the state (A(0) or B(1)) flux dataset (flux into state b = 1)
        flux_ab = np.array([expected[2] for expected in fluxes[:,self.state]])
        # CIs in rate (s^-1) format (divided by tau)
        ci_lb_ab = np.array([expected[3] for expected in fluxes[:,self.state]]) * (1/self.tau)
        ci_ub_ab = np.array([expected[4] for expected in fluxes[:,self.state]]) * (1/self.tau)

        # TODO: update to be a state assignment attr
        # norm by state pop A if calculating A --> B
        # if self.state == 1:
        #     state_pop = self.state_pop_a
        # # norm by state pop B if calculating B --> A
        # elif self.state == 0:
        #     state_pop = 1 - self.state_pop_a
        # # TODO: temp fix
        # else:
        #     state_pop = self.state_pop_a

        # assign the state of target flux flow
        if self.state == 1:
            state_pop = self.state_pop_a
        elif self.state == 0:
            state_pop = self.state_pop_b
        else:
            print("Currently only support state 0 or state 1.")

        # 2 different approaches here, can norm by state_pop_a (sum of weights in a)
        # but since 2 state system, could also use 1 - state_pop_b since all not in b are in a
        flux_ab = flux_ab / state_pop
        #flux_ab = flux_ab / state_pop_a
        #flux_ab = flux_ab / (1 - state_pop_b)

        # convert from tau^-1 to seconds^-1
        rate_ab = flux_ab * (1/self.tau)
        
        return rate_ab, ci_lb_ab, ci_ub_ab

    def plot_rate(self, title=None, moltime=True):
        """
        Plot the rate constant = target flux evolution AB / P_A 

        Returns
        -------
        rate_ab : ndarray
            Array of rates from A -> B in seconds^-1.
        """
        rate_ab, ci_lb_ab, ci_ub_ab = self.extract_rate()

        # WE iterations
        iterations = np.arange(0, len(rate_ab), 1)
        if moltime:
            # multiply by tau (ps)
            iterations *= 100
            # convert to ns
            iterations = np.divide(iterations, 1000)

        if self.units == "mfpts":
            mfpt_ab = 1 / rate_ab
            self.ax.plot(iterations, mfpt_ab, label=self.label)
            #ax.fill_between(iterations, mfpt_ab - (1/ci_lb_ab), mfpt_ab + (1/ci_ub_ab), alpha=0.5)
            self.ax.set_ylabel("MFPT ($s$)")
        elif self.units == "rates":
            self.ax.plot(iterations, rate_ab, color=self.color)#, label=self.label)
            self.ax.fill_between(iterations, rate_ab - ci_lb_ab, rate_ab + ci_ub_ab, alpha=0.5,
                                 label=self.label, color=self.color)
            self.ax.set_ylabel("Rate Constant ($s^{-1}$)")

        if moltime:
            self.ax.set_xlabel(r"Molecular Time (ns)")
        else:
            self.ax.set_xlabel(r"WE Iteration ($\tau$=100ps)")
        
        self.ax.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
        self.ax.set_title(title)

        return rate_ab

    def plot_statepop(self):
        """
        Plot the state populations
        """
        # # divide k_AB by P_A for equilibrium rate correction (AB and BA steady states)
        # state_pops = np.array(self.direct_h5["state_pop_evolution"])
        # state A = label 0, state B = label 1
        #state_pop_a = np.array([expected[2] for expected in self.state_pops[:,0]])
        #state_pop_b = np.array([expected[2] for expected in self.state_pops[:,1]])
        # state_pop_a = np.sum(self.state_pops[:,0], axis=1)
        # state_pop_b = np.sum(self.state_pops[:,1], axis=1)

        # WE iterations
        iterations = np.arange(0, len(self.state_pop_a), 1)

        # plot both state population evolutions
        self.ax.plot(iterations, self.state_pop_a, label="State A")
        self.ax.plot(iterations, self.state_pop_b, label="State B")
        self.ax.set_xlabel(r"WE Iteration ($\tau$=100ps)")
        self.ax.set_ylabel("State Population")

        #return self.state_pop_a, self.state_pop_b

    def plot_exp_vals(self, ax=None, f_range=False, d2d1=False, f_range_all=False):
        """
        f_range : bool
            Set to True to use mark 25-67 s^-1 as the k_D1D2 rate.
        d2d1 : bool
            Set to True to also include k_D2D1.
        """
        if ax is None:
            ax = self.ax
        if self.units == "rates":
            if f_range_all:
                # ax.axhline(60, alpha=1, color="tab:orange", label="4F k$_{D1D2}$", ls="--")
                # ax.axhline(25, alpha=1, color="tab:green", label="7F k$_{D1D2}$", ls="--")
                # ax.axhline(60, alpha=1, color="tab:red", ls="--")
                # ax.axhline(25, alpha=1, color="tab:green", ls="--")
                ax.axhline(60, alpha=1, color="tab:orange", ls="--")
                ax.axhline(25, alpha=1, color="tab:green", ls="--")
            elif f_range:
                # DTY 19F rates of 25-60 for k_D1D2
                ax.axhspan(25, 60, alpha=0.25, color="grey", label="NMR k$_{D1D2}$")
                if d2d1:
                    ax.axhspan(135, 179, alpha=0.25, color="tan", label="NMR k$_{D2D1}$")
            else:
                # D1-->D2 ~ 20-50, D2-->D1 ~ 100-150
                ax.axhline(150, color="k", ls="--", label="k$_{D2D1}$")
                if d2d1:
                    ax.axhline(25, color="red", ls="--", label="k$_{D1D2}$")
        elif self.units == "mfpts":
            # converted to mfpt = 1 / rate
            ax.axhline(1/150, color="k", ls="--", label="MFPT$_{D2D1}$")
            if d2d1:
                ax.axhline(1/25, color="red", ls="--", label="MFPT$_{D1D2}$")
        else:
            raise ValueError(f"You put {self.units} for unit, which must be `mfpts` or `rates`.") 

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="WEKAP: Weighted Ensemble Kinetics Analysis and Plotting"
    )
    parser.add_argument(
        "--direct", "--direct-h5",
        dest="direct",
        type=str,
        default="direct.h5",
        help="Name of output direct.h5 file from WESTPA w_direct or w_ipa."
    )
    parser.add_argument(
        "--assign", "--assign-h5",
        dest="assign",
        type=str,
        default=None,
        help="Name of specific assign.h5 file. Needed for labeled population data."
    )
    parser.add_argument(
        "--tau",
        dest="tau",
        type=float,
        default=100e-12,
        help="The resampling interval of the WE simulation in seconds. Default: 100ps = 100 * 10^-12 (s)."
    )
    parser.add_argument(
        "--state",
        dest="state",
        type=int,
        default=1,
        help="State for flux calculations (flux into `state`), 0 = A and 1 = B."
    )
    parser.add_argument(
        "--label",
        dest="label",
        type=str,
        default=None,
        help="Data label."
    )
    parser.add_argument(
        "--statepop",
        dest="statepop",
        type=str,
        default="direct",
        choices=["direct", "assign"],
        help="'direct' for state_population_evolution from direct.h5 or 'assign' for labeled_populations from assign.h5."
    )
    parser.add_argument(
        "--units",
        dest="units",
        type=str,
        default="rates",
        choices=["rates", "mfpts"],
        help="Measurement units. Can be `rates` (default) or `mfpts`."
    )
    parser.add_argument(
        "--savefig",
        dest="savefig",
        type=str,
        default=None,
        help="Path to optionally save the figure."
    )
    parser.add_argument(
        "--color",
        dest="color",
        type=str,
        default=None,
        help="Color of the line plot. Default None for mpl tab10 colors."
    )
    parser.add_argument(
        "--cumulative-avg", "-cavg",
        dest="cumulative_avg",
        action="store_true",
        help="Set to True when kinetics were calculated with cumulative averaging."
    )
    parser.add_argument(
        "--exp-values", "-exp",
        dest="exp_values",
        action="store_true",
        help="Plot experimental D1-->D2 values."
    )
    parser.add_argument(
        "--style",
        dest="style",
        type=str,
        default=None,
        help="Plot style. Can input path to mpl style file or use None or default."
    )
    return parser.parse_args()

if __name__ == "__main__":
    #fig, ax = plt.subplots()
    #k = Kinetics(f"D1D2_lt16oa/WT_v00/12oa/direct.h5", state=1, statepop="direct", ax=ax)
    args = parse_arguments()

    # plot style
    if args.style == "default":
        plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")
    elif args.style is not None:
        plt.style.use(args.style)

    # make plot
    k = Kinetics(**vars(args))
    k.plot_rate()

    # option to plot exp D1D2 values
    if args.exp_values:
        k.plot_exp_vals()
    
    plt.tight_layout()
    # option to save figure output
    if args.savefig is not None:
        plt.savefig(args.savefig)
    
    plt.show()
