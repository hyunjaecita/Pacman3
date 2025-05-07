from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions(agentIndex):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
        Implementación del agente MiniMax para Pacman,
        este actúa como MAX, el fantasma como MIN.
        Se utiliza profundidad limitada (self.depth) y evaluación heurística.
    """

    def getAction(self, gameState: GameState) -> str:

        # Definimos constantes para la identificación de los agentes
        PACMAN = 0
        GHOST = 1

        def minimax(state, depth, agentIndex):
            """
                Función recursiva principal del algoritmo MiniMax.
                Alterna entre funciones de maximización (Pacman) y minimización (Fantasma),
                controlando la profundidad.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == PACMAN:
                return max_value(state, depth)
            else:
                return min_value(state, depth)


        def max_value(state, depth):
            """
                Función MAX para el turno de Pacman.
                Busca la acción que maximiza el resultado esperado.
            """
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf") # Inicializamos al peor caso posible
            best_action = Directions.STOP

            for action in actions:
                # Generamos el estado sucesor si Pacman realiza esta acción
                successor = state.generateSuccessor(PACMAN, action)
                score = minimax(successor, depth, GHOST)  # Llamamos a minimax para el siguiente agente: el fantasma

                # Si el valor obtenido es mejor, actualiza el mejor
                if score > best_score:
                    best_score = score
                    best_action = action

            if depth == 0:  # Si estamos en el primer nivel de profundidad, devolvemos la acción
                return best_action
            else:  # En otros niveles, devolvemos la mejor puntuación
                return best_score

        def min_value(state, depth):
            """
               Función MIN para el turno del fantasma.
               Busca la acción que minimiza la ganancia de Pacman.
            """
            best_score = float("inf") # Inicializamos al mejor caso para MIN (peor para Pacman)
            actions = state.getLegalActions(GHOST)

            for action in actions:
                # Generamos el estado sucesor si el fantasma realiza esta acción
                successor = state.generateSuccessor(GHOST, action)
                # Ahora es turno de Pacman, por lo tanto aumentamos la profundidad
                score = minimax(successor, depth + 1, PACMAN)

                if score < best_score:
                    best_score = score

            return best_score

        return max_value(gameState, 0)



######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Agente que implementa el algoritmo MiniMax con poda Alpha-Beta.
    """

    def getAction(self, gameState: GameState) -> str:
        """
            Devuelve la mejor acción para Pacman usando MiniMax con poda Alpha-Beta.
        """
        PACMAN = 0
        GHOST = 1

        def alphabeta(state, depth, agentIndex, alpha, beta):
            """
                Función principal de búsqueda recursiva Alpha-Beta.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == PACMAN:
                return max_value(state, depth, alpha, beta)
            else:
                return min_value(state, depth, alpha, beta)

        def max_value(state, depth, alpha, beta):
            """
                Función para el turno de Pacman (nodo MAX).
                Elige la acción que maximiza el valor, y realiza poda cuando beta <= alpha.
            """
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            best_action = Directions.STOP

            for action in actions:
                successor = state.generateSuccessor(PACMAN, action) # Generamos el estado sucesor
                score = alphabeta(successor, depth, GHOST, alpha, beta) # Llamada recursiva con el turno del fantasma

                if score > best_score:
                    best_score = score
                    best_action = action

                # Actualizamos alpha
                alpha = max(alpha, best_score)

                # Poda: si el valor actual ya es mayor o igual que beta, no seguimos explorando
                if alpha >= beta:
                    break

            if depth == 0:
                return best_action
            else:
                return best_score

        def min_value(state, depth, alpha, beta):
            """
                Función para el turno del fantasma (nodo MIN).
                Elige la acción que minimiza el valor, y realiza poda cuando beta <= alpha.
            """
            actions = state.getLegalActions(GHOST)
            best_score = float("inf")

            for action in actions:
                successor = state.generateSuccessor(GHOST, action)
                score = alphabeta(successor, depth + 1, PACMAN, alpha, beta) # Turno vuelve a Pacman, profundidad +1

                if score < best_score:
                    best_score = score

                # Actualizamos beta
                beta = min(beta, best_score)

                # Poda: si el valor actual ya es mayor o igual que beta, no seguimos explorando
                if beta <= alpha:
                    break

            return best_score

        return alphabeta(gameState, 0, PACMAN, float('-inf'), float('inf'))


######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
        Agente basado en el algoritmo Expectimax (Problema 3).
        Pacman actúa como nodo MAX (elige la mejor opción),
        y los fantasmas como nodos de expectativa (eligen aleatoriamente).
    """

    def getAction(self, gameState: GameState) -> str:
     """
       Devuelve la acción que Pacman debe realizar usando búsqueda Expectimax
       hasta la profundidad indicada en self.depth.
       Los fantasmas se modelan como agentes que eligen sus movimientos
       al azar (uniformemente).
     """
     PACMAN = 0
     GHOST = 1

     def expectimax(state, depth, agentIndex):
         """
           Función principal de búsqueda recursiva Expectimax.
           Cambia entre MAX (Pacman) y EXP (fantasmas).
         """
         if state.isWin() or state.isLose() or depth == self.depth:
             return self.evaluationFunction(state)

         if agentIndex == PACMAN:
             return max_value(state, depth)
         else:
             return exp_value(state, depth)

     def max_value(state, depth):
         """
             Turno de Pacman: nodo MAX. Se escoge la acción con el mayor valor esperado.
         """
         actions = state.getLegalActions(PACMAN)
         best_score = float("-inf")
         best_action = Directions.STOP

         for action in actions:
             successor = state.generateSuccessor(PACMAN, action)
             score = expectimax(successor, depth, GHOST)

             if score > best_score:
                 best_score = score
                 best_action = action

         if depth == 0:
             return best_action
         else:
             return best_score

     def exp_value(state, depth):
         """
           Función para el turno de los fantasmas (nodo EXPECTATION).
           Calcula el valor esperado suponiendo elección uniforme entre acciones legales.
         """

         actions = state.getLegalActions(GHOST)
         total_score = 0
         prob = 1.0 / len(actions)  # Asumimos distribución uniforme

         for action in actions:
             successor = state.generateSuccessor(GHOST, action)
             total_score += prob * expectimax(successor, depth + 1, PACMAN)  # Turno vuelve a Pacman

         return total_score  # Valor esperado de todas las acciones posibles del fantasma

     return expectimax(gameState, 0, PACMAN)

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
      Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
    """

# Abbreviation
better = betterEvaluationFunction
