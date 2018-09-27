import copy
import numpy as np
from numpy import linalg as LA

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        self.debug = False

    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        #check some input variables
        assert initialV.ndim == 1, "Invalid initialV: it has dimensionality " + repr(initialV.ndim)
        assert initialV.shape[0] == self.nStates, "Invalid initialV shape: it has shape " + repr(initialV.shape)
        #initialize variables for value iteration
        V_star = initialV  #using initialV as it was given in slide 15 (2b) that all initial estimates
                           #for value iteration will terminate (just with different number of iterations)
        iterId = 0
        epsilon = 0.
        changeInV = True
        V_act = np.zeros([self.nActions, self.nStates])
        #loop until V stops changing or epsilon <= tolerance
        while changeInV:
            for act_idx in range(self.nActions):
                if self.debug:
                    print ("[DEBUG:VI] R[act_idx].shape: {}".format(self.R[act_idx].shape))
                    print ("[DEBUG:VI] T[act_idx].shape: {}".format(self.T[act_idx].shape))
                    print ("[DEBUG:VI] V_star.shape: {}".format(V_star.shape))
                    print ("[DEBUG:VI] right term shape: {}".format(np.matmul(self.T[act_idx], V_star).shape))
                    print ("[DEBUG:VI] full term shape: {}".format((self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_star))).shape))
                #for each action, compute the V and then select V_star based on max of element wise
                V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_star))
            iterId += 1
            epsilon = LA.norm(np.subtract(V_star,np.amax(V_act, axis=0)), np.inf)
            if self.debug:
                print ("[DEBUG:VI] V_act[0] max {}".format(np.amax(V_act, axis=0)))
                print ("[DEBUG:VI] V_star {}".format(V_star))
                print ("[DEBUG:VI] epsilon: {}".format(epsilon))
            #convergence condition
            if (tolerance == 0. and epsilon == 0. ) or \
               (tolerance > 0. and epsilon <= tolerance) or \
               (nIterations != np.inf and iterId == nIterations):
                #the algorithm will only converge when the value function for states stops changing OR
                #the algorithm will converge when the change in value function is less than tolerance
                # (based on the comment on piazza that stop the algo when the epsilon <= tolerance) OR
                #the algorithm will converge when the nIterations have expired
                changeInV = False
            V_star = np.amax(V_act, axis=0)

        V = V_star
        if self.debug:
            print ("[DEBUG:VI] V: {}".format(V))
            print ("[DEBUG:VI] iterId: {}".format(iterId))
            print ("[DEBUG:VI] epsilon: {}".format(epsilon))

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        assert V.ndim == 1, "Invalid V: it has dimensionality " + repr(V.ndim)
        assert V.shape[0] == self.nStates, "Invalid V: it has shape " + repr(V.shape)

        #the policy can be extracted by running the value iteration step one more time
        #because the value iteration has converged, we should have a stationary policy, so
        #running the value iteration step one more time won't change the policy
        V_act = np.zeros([self.nActions, self.nStates])
        for act_idx in range(self.nActions):
            if self.debug:
                print ("[DEBUG:ExP] right term shape: {}".format(np.matmul(self.T[act_idx], V).shape))
                print ("[DEBUG:ExP] full term shape: {}".format((self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V))).shape))
            #for each action, compute the V and then select V based on max of element wise
            V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V))
        if self.debug:
            print ("[DEBUG:ExP] V_act amax {}".format(np.amax(V_act, axis=0)))
            print ("[DEBUG:ExP] V_act argmax {}".format(np.argmax(V_act, axis=0)))
        policy = np.argmax(V_act, axis=0)

        return policy

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        #get the reward signal and transition matrix for the current policy
        R_pi = np.zeros([self.nStates])
        T_pi = np.zeros([self.nStates,self.nStates], dtype=np.float)
        V_pi = np.zeros([self.nStates])
        changeInV = True
        for state_idx in range(self.nStates):
            action = policy[state_idx]
            R_pi[state_idx] = self.R[action, state_idx]
            T_pi[state_idx] = self.T[action, state_idx, :]
        if self.debug:
            print ("[DEBUG:EvP] policy: {}".format(policy))
            print ("[DEBUG:EvP] R_pi: {}".format(R_pi))
            print ("[DEBUG:EvP] T_pi: {}".format(T_pi))
            print ("[DEBUG:EvP] self.T: {}".format(self.T))
            print ("[DEBUG:EvP] R_pi shape: {}".format(R_pi.shape))
            print ("[DEBUG:EvP] T_pi shape: {}".format(T_pi.shape))
            print ("[DEBUG:EvP] T shape: {}".format(self.T.shape))
            print ("[DEBUG:EvP] V_pi shape: {}".format(V_pi.shape))
            print ("[DEBUG:EvP] right term: {}".format(np.matmul(T_pi, V_pi).shape))
        while changeInV:
            V_new = R_pi + (self.discount*np.matmul(T_pi, V_pi))
            #full policy evaluation requires iterations until the value function stops changing or epsilon==0
            if np.array_equal(V_new, V_pi):
                changeInV = False
            V_pi = V_new
        V = V_pi

        return V

    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        #sanity checking
        assert initialPolicy.ndim == 1, "Invalid initialPolicy: it has dimensionality " + repr(initialPolicy.ndim)
        assert initialPolicy.shape[0] == self.nStates, "Invalid initialPolicy: it has shape " + repr(initialPolicy.shape)
        # loop till the policy stops updates
        changeInP = True
        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0
        while changeInP:
            #evaluate policy
            V_eval = self.evaluatePolicy(policy)
            #generate new policy using V_eval
            V_act = np.zeros([self.nActions, self.nStates])
            for act_idx in range(self.nActions):
                V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_eval))
            policy_new = np.argmax(V_act, axis=0)
            if self.debug:
                print ("[DEBUG PI] V_eval: {}".format(V_eval))
                print ("[DEBUG PI] policy_new: {}".format(policy_new))
            if np.array_equal(policy_new, policy) or np.array_equal(V_eval, V) or \
               (nIterations != np.inf and iterId == nIterations):
                #from lecture 3a video 14:00, we should also have stopping condition where the value function is same
                changeInP = False
            policy = policy_new
            V = V_eval
            iterId += 1

        return [policy,V,iterId]

    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        #get the reward signal and transition matrix for the current policy
        R_pi = np.zeros([self.nStates])
        T_pi = np.zeros([self.nStates,self.nStates], dtype=np.float)
        V_pi = initialV
        changeInV = True
        iterId = 0
        epsilon = 0.
        for state_idx in range(self.nStates):
            action = policy[state_idx]
            R_pi[state_idx] = self.R[action, state_idx]
            T_pi[state_idx] = self.T[action, state_idx]
        if self.debug:
            print ("[DEBUG EvPp] policy: {}".format(policy))
            print ("[DEBUG EvPp] R_pi: {}".format(R_pi))
            print ("[DEBUG EvPp] T_pi: {}".format(T_pi))
            print ("[DEBUG EvPp] R_pi shape: {}".format(R_pi.shape))
            print ("[DEBUG EvPp] T_pi shape: {}".format(T_pi.shape))
            print ("[DEBUG EvPp] T shape: {}".format(self.T.shape))
            print ("[DEBUG EvPp] V_pi shape: {}".format(V_pi.shape))
            print ("[DEBUG EvPp] right term: {}".format(np.matmul(T_pi, V_pi).shape))
        if nIterations == 0:
            changeInV = False
        #repeat richardson iteration for nIterations or until epsilon is converged
        while changeInV:
            V_new = R_pi + (self.discount*np.matmul(T_pi, V_pi))
            iterId += 1
            epsilon = LA.norm(np.subtract(V_new,V_pi), np.inf)
            if (tolerance == 0. and epsilon == 0. ) or \
               (tolerance > 0. and epsilon <= tolerance) or \
               (nIterations != np.inf and iterId == nIterations):
                changeInV = False
            V_pi = V_new
        V = V_pi

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01, report_full_iter=False):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)
        report_full_iter (False) -- report the total number of iterations for policy iteration + partial policy evaluation

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        #sanity checking
        assert initialPolicy.ndim == 1, "Invalid initialPolicy: it has dimensionality " + repr(initialPolicy.ndim)
        assert initialPolicy.shape[0] == self.nStates, "Invalid initialPolicy: it has shape " + repr(initialPolicy.shape)
        assert initialV.ndim == 1, "Invalid initialV: it has dimensionality " + repr(initialV.ndim)
        assert initialV.shape[0] == self.nStates, "Invalid initialV shape: it has shape " + repr(initialV.shape)
        # loop till the policy stops updates or we reach tolerance
        changeInP = True
        policy = initialPolicy
        V = initialV
        V_next = initialV
        iterId = 0
        epsilon = 0.
        while changeInP:
            V_eval, eval_iter, eval_epsilon = self.evaluatePolicyPartially(policy, V_next, nIterations=nEvalIterations, tolerance=tolerance)
            if report_full_iter:
                iterId += eval_iter
            V_act = np.zeros([self.nActions, self.nStates])
            for act_idx in range(self.nActions):
                V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_eval))
            policy_new = np.argmax(V_act, axis=0)
            V_next = np.amax(V_act, axis=0)
            if self.debug:
                print ("[DEBUG:MPI] V_eval: {}".format(V_eval))
                print ("[DEBUG:MPI] policy_new: {}".format(policy_new))
            iterId += 1
            epsilon = LA.norm(np.subtract(V_eval, V_next), np.inf)
            if (tolerance == 0. and epsilon == 0.) or \
               (tolerance > 0. and epsilon <= tolerance) or \
               (nIterations != np.inf and nIterations == iterId):
               #convergence conditions are similar to value iteration where the change in value iteration
               #should be less than epsilon or run at least for nIterations
                changeInP = False
            policy = policy_new
            V = V_eval


        return [policy,V,iterId,epsilon]

