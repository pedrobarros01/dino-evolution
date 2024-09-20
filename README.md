# Algoritmo Genético para Agente do Jogo Dino

Este README fornece uma visão geral da implementação do algoritmo genético para treinar um agente do jogo Dino. Ele cobre o fluxo geral, abordagens utilizadas para gerar soluções, métodos de seleção, mutação e as funções de ativação usadas no código.

## Visão Geral

O objetivo do algoritmo genético é evoluir uma população de agentes para jogar o jogo Dino de forma eficaz. Para os agentes foram utilizados Perceptrons e futuramente na solução final Redes Neurais para decidir as ações com base no estado do jogo. Algoritmos genéticos comumente envolvem os seguintes passos:

1. **Inicialização:** Criação de uma população de agentes com pesos e viéses aleatórios.
2. **Avaliação:** Testar o desempenho de cada agente no jogo Dino e atribuir uma pontuação de aptidão com base no desempenho.
3. **Seleção:** Escolher os melhores agentes para serem pais da próxima geração.
4. **Crossover:** Combinar os genes (pesos e viéses) dos agentes pais para produzir descendentes.
5. **Mutação:** Introduzir alterações aleatórias nos genes dos descendentes para manter a diversidade genética.
6. **Repetição:** Continuar o processo por um número específico de gerações para evoluir melhores agentes.

## Componentes do Código de Cada Solução

## Perceptron + Algoritmo Genético (Primeira solução)

### 1. `process_game_state(game_state)`

Processa o estado do jogo em um formato simplificado para o agente. Esta função extrai informações relevantes, como a posição do dinossauro, posições dos obstáculos e velocidade do jogo. Retorna uma lista de valores que representam o estado do jogo.

### 2. `reward_function(current_score, previous_score, state)`

Calcula a recompensa para um agente com base no aumento de pontuação e na distância percorrida. Isso incentiva os agentes a melhorar sua pontuação e percorrer mais distância no jogo.

### 3. `evaluate_agent(agents, game)`

Avalia cada agente simulando o jogo. Retorna uma lista de agentes com suas pontuações correspondentes. As pontuações são calculadas usando a função de recompensa.

### 4. `genetic_algorithm(generations, population_size, mutation_rate, elitism_size)`

A função principal para rodar o algoritmo genético. Inicializa uma população de agentes, avalia seu desempenho, realiza seleção, crossover e mutação, e itera pelo número especificado de gerações. O melhor agente é retornado após todas as gerações.

### 5. `evaluate_agent_performance(agent, num_trials)`

Testa o desempenho de um único agente em múltiplas tentativas. Retorna a pontuação média do agente, fornecendo uma estimativa de seu desempenho geral.

## Rede Neural + Algoritmo Genético (Segunda solução)

### Classe NeuralNetworkAgent

Representa o agente com uma rede neural de uma camada oculta.

- **init(self, input_size, hidden_size)**: Inicializa os pesos da rede neural.

  - input_size: Tamanho da entrada da rede (estado do jogo).
  - hidden_size: Número de neurônios na camada oculta.

- **relu(self, x)**: Função de ativação ReLU. Retorna o valor máximo entre 0 e x, introduzindo não linearidade na rede.

- **get_action(self, state)**:

  - Realiza uma passagem direta pela rede neural (input -> camada oculta -> saída).
  - Retorna ACTION_UP se a saída for maior que 0, e ACTION_DOWN caso contrário, controlando as ações do agente.

- **mutate(self, mutation_rate=0.05, mutation_intensity=1000)**:

  - Aplica mutações aleatórias nos pesos da rede.
  - Para cada peso na camada de entrada para a camada oculta, se um número aleatório for menor que mutation_rate, o peso é alterado.
  - O mesmo processo é aplicado para os pesos da camada oculta para a saída.

### Função evaluate_agent(agents, game)

Avalia uma lista de agentes no jogo.

- Reseta o estado do jogo.
- Executa um loop enquanto o jogo não estiver encerrado, obtendo o estado atual do jogo e as ações dos agentes.
- Retorna as pontuações de cada agente, associadas a seus respectivos objetos.

## Detalhes das soluções

### Seleção

A seleção é o processo de escolher os melhores agentes para serem pais da próxima geração:

- **Primeira solução - Elitismo:** Um número de agentes de topo (definido por `elitism_size`) é diretamente transferido para a próxima geração e para gerar descendentes. Isso garante que as melhores soluções sejam preservadas.

- **Segunda solução - CLone:** O melhor indivíduo da solução passada é clonado e todos os indivíduo da geração seguinte se tornam réplicas dele.

### Crossover

O crossover combina o material genético (pesos e viéses) de dois agentes pais para criar descendentes, esta abordagem foi utilizada __apenas na primeira solução__, após diversos testes e com diferentes crossover (mostrados abaixo) foi concluído que para este cenário não seria proveitoso:

- **Crossover Uniforme:** Seleciona aleatoriamente genes de qualquer um dos pais para criar o descendente. Cada gene no descendente tem a mesma probabilidade de vir de qualquer um dos pais.

  ```python
  @classmethod
  def crossover_uniforme(cls, parent1, parent2):
      new_weights = np.array(
          [
              [
                  random.choice([w1, w2]) for w1, w2 in zip(row1, row2)
              ]  # Crossover uniforme
              for row1, row2 in zip(parent1.weights, parent2.weights)
          ]
      )
      new_bias = [
          random.choice([b1, b2]) for b1, b2 in zip(parent1.bias, parent2.bias)
      ]
      return cls(new_weights, parent1.activation_function, new_bias)
  ```

- **Crossover de Ponto Único:** Escolhe um ponto aleatório nos genes e combina genes dos dois pais até esse ponto de um pai e o restante do outro pai. Este método pode ser mais simples e pode preservar características específicas dos pais.

  ```python
  @classmethod
  def crossover_unico(cls, parent1, parent2):
      point = random.randint(0, len(parent1.weights) - 1)
      new_weights = np.array(
          [
              row1 if i < point else row2
              for i, (row1, row2) in enumerate(zip(parent1.weights, parent2.weights))
          ]
      )
      new_bias = parent1.bias[:point] + parent2.bias[point:]
      return cls(new_weights, parent1.activation_function, new_bias)
  ```

- **Crossover Blend:** Faz uma média ponderada dos genes dos pais, permitindo transições suaves entre os genes dos pais. Isso pode promover uma evolução mais estável e gradual.

  ```python
  @classmethod
  def crossover_blend(cls, parent1, parent2, alpha=0.5):
      new_weights = np.array(
          [
              [(alpha * w1 + (1 - alpha) * w2) for w1, w2 in zip(row1, row2)]
              for row1, row2 in zip(parent1.weights, parent2.weights)
          ]
      )
      new_bias = [
          (alpha * b1 + (1 - alpha) * b2)
          for b1, b2 in zip(parent1.bias, parent2.bias)
      ]
      return cls(new_weights, parent1.activation_function, new_bias)
  ```

### Funções de Ativação

As funções de ativação são usadas para calcular a saída do perceptron e pode introduzir não-linearidades nestas soluções, foram testadas diferentes funções de ativação ao longo do desenvolvimento do projeto:

- **ReLU (Rectified Linear Unit):** Transforma valores negativos em zero e mantém valores positivos. É simples e eficiente, mas pode sofrer do problema de "morte de neurônios" se muitos valores forem negativos.

  ```python
  def relu(self, x):
    return np.maximum(0, x)
  ```

- **Sigmoid:** Mapeia valores para o intervalo (0, 1). É útil para problemas de classificação, mas pode sofrer de gradientes muito pequenos, o que dificulta o treinamento (problema de desaparecimento do gradiente).

  ```python
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  ```

- **Tanh (Tangente Hiperbólica):** Mapeia valores para o intervalo (-1, 1). É uma versão centrada da sigmoid, o que pode ajudar a melhorar o desempenho em alguns casos, mas ainda pode sofrer do problema de desaparecimento do gradiente.

  ```python
  def tanh(self, x):
    return np.tanh(x)
  ```

- **Leaky ReLU:** Permite uma pequena inclinação para valores negativos, o que pode ajudar a mitigar o problema de morte de neurônios, mantendo a capacidade de aprender padrões.

  ```python
  def leaky_relu(self, x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)
  ```

- **Softmax:** Converte os valores de saída em probabilidades. É utilizado principalmente para problemas de classificação multi-classe, onde você precisa obter probabilidades para diferentes classes.

  ```python
  def softmax(self, x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)
  ```

- **ELU (Exponential Linear Unit):** Adiciona uma exponencial para valores negativos, o que pode acelerar o treinamento e reduzir o problema de morte de neurônios, mantendo valores negativos em uma forma útil.

  ```python
  def elu(self, x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
  ```

## Best Agent

Para obter o best agente ao fim do treinamento, os pesos são salvos em um arquivo. Futuramente para testes com o melhor agente obtido as informações são carregadas deste arquivo e carregadas novamente utilizando a classe NeuralNetworkAgent.

- Cria uma instância de DinoGame para iniciar o jogo.
- Carrega os pesos do melhor agente do arquivo JSON (best_agent.json).
- Executa um loop até que o jogo termine, onde o agente toma ações baseadas no estado atual.
- Se a pontuação atingir 30.000, o jogo é encerrado.
- Exibe a pontuação final após o término do jogo.