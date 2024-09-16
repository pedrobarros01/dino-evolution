# python-repository-template

A Python repository template to facilitate getting your projects started and organized.

# If you use Windows, use chocolatey for installing things

- [chocolatey installation guide](https://chocolatey.org/install)

# Use pyenv for Python version management

- [pyenv installation guide](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)

```bash
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
exit
```

In another shell:

```bash
pyenv update
pyenv install 3.9.13
pyenv rehash
pyenv global 3.9.13
exit
```

# Use `make` for simplifing commands and making it explicit how to run your code

- [make documentation](https://www.gnu.org/software/make/manual/make.html)

# Use poetry for managing Python dependencies

[Poetry](https://python-poetry.org/docs/basic-usage/) is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry offers a lockfile to ensure repeatable installs, and can build your project for distribution.

## Basic commands:

- Add new dependency: `poetry add <package>`
- Install dependencies: `poetry install`
- Update dependencies: `poetry update`
- Remove dependencies: `poetry remove <package>`
- Run a command in the virtual environment: `poetry run <command>`
- Run python in the virtual environment: `poetry run python <command>`

# Make sure to use the Makefile to facilitate the usage of your repository

Anyone that clones your repository should be able to relatively easily run your code with just a few commands. The Makefile should contain the following commands:

```bash
make install
make run
```

# Use pre-commit for running checks before committing

[pre-commit](https://pre-commit.com/) is a framework for managing and maintaining multi-language pre-commit hooks. It is a client-side hook manager that can be used to automate checks before committing code. It is recommended to use pre-commit to ensure code formatting, among other things.

# Algoritmo Genético para Agente do Jogo Dino

Este README fornece uma visão geral da implementação do algoritmo genético para treinar um agente do jogo Dino. Ele cobre o fluxo geral, métodos de seleção, mutação e as funções de ativação usadas no código.

## Visão Geral

O objetivo do algoritmo genético é evoluir uma população de agentes para jogar o jogo Dino de forma eficaz. Os agentes utilizam Perceptron para decidir as ações com base no estado do jogo. O algoritmo envolve os seguintes passos:

1. **Inicialização:** Criação de uma população de agentes com pesos e viéses aleatórios.
2. **Avaliação:** Testar o desempenho de cada agente no jogo Dino e atribuir uma pontuação de aptidão com base no desempenho.
3. **Seleção:** Escolher os melhores agentes para serem pais da próxima geração.
4. **Crossover:** Combinar os genes (pesos e viéses) dos agentes pais para produzir descendentes.
5. **Mutação:** Introduzir alterações aleatórias nos genes dos descendentes para manter a diversidade genética.
6. **Repetição:** Continuar o processo por um número específico de gerações para evoluir melhores agentes.

## Componentes do Código

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

## Detalhes do Algoritmo Genético

### Seleção

A seleção é o processo de escolher os melhores agentes para serem pais da próxima geração:

- **Elitismo:** Um número de agentes de topo (definido por `elitism_size`) é diretamente transferido para a próxima geração e para gerar descendentes. Isso garante que as melhores soluções sejam preservadas.

### Crossover

O crossover combina o material genético (pesos e viéses) de dois agentes pais para criar descendentes:

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

As funções de ativação são usadas para calcular a saída do perceptron e pode introduzir não-linearidades nestas soluções:

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

## Execução do Algoritmo

Para executar o algoritmo genético os parâmetros ajustáveis são:

- **generations:** Número de gerações para evolução.
- **population_size:** Número de agentes em cada geração.
- **mutation_rate:** Probabilidade de mutação ocorrer em um agente.
- **elitism_size:** Número de agentes de topo a gerarem descendentes e serem preservados para a próxima geração.

  ```python
  if __name__ == "__main__":
    generations = 25
    population_size = 50
    mutation_rate = 0.1
    elitism_size = 5

    best_agent = genetic_algorithm(generations, population_size, mutation_rate, elitism_size)
    evaluate_agent_performance(best_agent)
  ```