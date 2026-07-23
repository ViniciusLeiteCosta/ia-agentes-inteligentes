import time

from labirinto import Labirinto
from agentes import AgenteAEstrela, AgenteQLearning, AgenteGenetico

CENARIOS = [
    {"semente": 1, "indice_partida": 0},
    {"semente": 2, "indice_partida": 1},
    {"semente": 3, "indice_partida": 2},
    {"semente": 17, "indice_partida": 3},
    {"semente": 42, "indice_partida": 0},
]

REPETICOES_POR_CENARIO = 3  # Q-Learning e Genético têm componente aleatório


def rodar_agente(agente, labirinto, partida, max_passos):
    inicio = time.perf_counter()
    agente.preparar_rodada(labirinto, partida)
    passos = 0
    while not agente.concluido and passos < max_passos:
        agente.passo(labirinto)
        passos += 1
    duracao = time.perf_counter() - inicio
    return agente.concluido, agente.passos_dados, duracao


def avaliar():
    resultados = {"A*": [], "Q-Learning": [], "Genético": []}

    for cenario in CENARIOS:
        labirinto = Labirinto(tamanho=17, conexoes_extras=0.15, semente=cenario["semente"])
        partida = labirinto.pontos_partida[cenario["indice_partida"]]
        distancia_bfs = labirinto.distancias[partida]  # limite inferior teórico
        max_passos = labirinto.tamanho * 8

        agente_astar = AgenteAEstrela((220, 40, 40))
        agente_rl = AgenteQLearning((30, 100, 220))
        agente_ga = AgenteGenetico((30, 160, 60))

        # Q-Learning precisa treinar uma vez para cada labirinto novo (a
        # tabela Q aprendida é específica daquele mapa).
        agente_rl.treinar(labirinto, labirinto.pontos_partida)

        for _ in range(REPETICOES_POR_CENARIO):
            sucesso, passos, tempo = rodar_agente(agente_astar, labirinto, partida, max_passos)
            resultados["A*"].append((sucesso, passos, tempo, distancia_bfs))

            sucesso, passos, tempo = rodar_agente(agente_rl, labirinto, partida, max_passos)
            resultados["Q-Learning"].append((sucesso, passos, tempo, distancia_bfs))

            sucesso, passos, tempo = rodar_agente(agente_ga, labirinto, partida, max_passos)
            resultados["Genético"].append((sucesso, passos, tempo, distancia_bfs))

    return resultados


def resumir(resultados):
    print(f"{'Agente':<12} {'Sucesso':>9} {'Passos (med.)':>15} {'Tempo (med. s)':>16} {'BFS (med.)':>12}")
    for nome, execucoes in resultados.items():
        n = len(execucoes)
        sucessos = sum(1 for s, _, _, _ in execucoes if s)
        media_passos = sum(p for s, p, _, _ in execucoes if s) / max(sucessos, 1)
        media_tempo = sum(t for _, _, t, _ in execucoes) / n
        media_bfs = sum(b for _, _, _, b in execucoes) / n
        taxa_sucesso = 100 * sucessos / n
        print(f"{nome:<12} {taxa_sucesso:>8.0f}% {media_passos:>15.1f} {media_tempo:>16.3f} {media_bfs:>12.1f}")


if __name__ == "__main__":
    resultados = avaliar()
    resumir(resultados)