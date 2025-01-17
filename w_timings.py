import numpy as np
import h5py
import argparse

from tqdm.auto import tqdm

class WTimings:
    """
    Aggregate simulation and wallclock time extraction.

    TODO:
        * convert to use logging
        * integrate into WESTPA, pull instance attributes from west.cfg (e.g. west.h5 filename)
    """

    def __init__(self, h5="west.h5", tau=100, first_iter=1, last_iter=None, count_events=False):
        """
        Parameters
        ----------
        tau : int
            WESTPA dynamics propagation time in picoseconds. Default 100 = 100ps.
        h5 : str
            Path to west.h5 file
        first_iter : int
            By default start at iteration 1.
        last_iter : int
            Last iteration data to include, default is the last recorded iteration in the west.h5 file. 
        count_events : bool
            Option to also output the number of successfull recycling events.
        """
        self.tau = tau
        self.h5 = h5py.File(h5, mode="r")
        self.first_iter = int(first_iter)
        # default to last
        if last_iter is not None:
            self.last_iter = int(last_iter)
        elif last_iter is None:
            self.last_iter = self.h5.attrs["west_current_iteration"] - 1
        self.count_events = count_events

    def get_event_count(self):
        """
        Check if the target state was reached, given the data in a WEST H5 file.

        Returns
        -------
        int
            Number of successful recycling events.
        """
        events = 0
        # Get the key to the final iteration. 
        # Need to do -2 instead of -1 because there's an empty-ish final iteration written.
        for iteration_key in tqdm(list(self.h5['iterations'].keys())[-2:0:-1]):
            endpoint_types = self.h5[f'iterations/{iteration_key}/seg_index']['endpoint_type']
            if 3 in endpoint_types:
                #print(f"recycled segment found in file {h5_filename} at iteration {iteration_key}")
                # count the number of 3s
                events += np.count_nonzero(endpoint_types == 3)
        return events

    def main(self):
        """
        Public class method, get timings.
        """
        walltime = self.h5['summary']['walltime'][self.first_iter-1:self.last_iter].sum()
        aggtime = self.h5['summary']['n_particles'][self.first_iter-1:self.last_iter].sum()

        print("\nwalltime: ", walltime, "seconds")
        print("walltime: ", walltime/60, "minutes")
        print("walltime: ", walltime/60/60, "hours")
        print("walltime: ", walltime/60/60/24, "days")
        print(f"\nassuming tau of {self.tau} ps:")
        print("aggtime: ", aggtime, "segments ran for tau intervals")
        print("aggtime: ", (aggtime * self.tau)/1000, "ns")
        print("aggtime: ", (aggtime * self.tau)/1000/1000, "µs\n")
        if self.count_events:
            print("successful recycling events:", self.get_event_count(), "\n")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="w_timings: a tool for aggregate simulation and wallclock time extraction."
    )
    parser.add_argument(
        "-W", "-w", "--west", "--west-data", "-h5", "--h5file",
        dest="h5",
        type=str,
        default="west.h5",
        help="Path to west.h5 file"
    )
    parser.add_argument(
        "--tau", "-t",
        dest="tau",
        type=int,
        default=100,
        help="WESTPA dynamics propagation time in picoseconds. Default 100 = 100ps."
    )
    parser.add_argument(
        "--first-iter", "-fi",
        dest="first_iter",
        type=int,
        default=1,
        help="First iteration to consider (default: 1)"
    )
    parser.add_argument(
        "--last-iter", "-li",
        dest="last_iter",
        type=int,
        default=None,
        help="Last iteration to consider (default: last recorded iteration in west.h5)"
    )
    parser.add_argument(
        "--count-events", "-ce",
        dest="count_events",
        action="store_true",
        default=False,
        help="Include this flag to also output the number of successfull recycling events."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    timings = WTimings(**vars(args))
    timings.main()