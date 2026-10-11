"""Exercícios da lição 09 - Data Management.

Uso (com o .venv ativo):
    python exercicios_data_management.py        # roda os 4
    python exercicios_data_management.py 3      # roda só o exercício 3

Antes, mande o cache do Hugging Face para o HD:
    export HF_HOME=/media/nickzs/D/cache/huggingface
"""
import os
import sys
import time

from datasets import load_dataset


def exercicio_1():
    """Carregar o glue (config mrpc) e ver os 5 primeiros exemplos."""
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    print(dataset)

    for i in range(5):
        print(dataset["train"][i])


def exercicio_2():
    """Streaming do c4: quantos exemplos em 10 segundos."""
    stream = load_dataset("allenai/c4", "en", split="train", streaming=True)

    inicio = time.time()
    contador = 0
    for _ in stream:
        contador += 1
        if time.time() - inicio >= 10:
            break

    print(f"Exemplos processados em 10 segundos: {contador}")


def exercicio_3():
    """Converter para Parquet e comparar o tamanho com o CSV."""
    dataset = load_dataset("stanfordnlp/imdb", split="train")

    os.makedirs("saida", exist_ok=True)
    caminho_csv = "saida/imdb_train.csv"
    caminho_parquet = "saida/imdb_train.parquet"

    dataset.to_csv(caminho_csv)
    dataset.to_parquet(caminho_parquet)

    csv_mb = os.path.getsize(caminho_csv) / 1_000_000
    parquet_mb = os.path.getsize(caminho_parquet) / 1_000_000

    print(f"CSV:     {csv_mb:.1f} MB")
    print(f"Parquet: {parquet_mb:.1f} MB")
    print(f"O Parquet tem {parquet_mb / csv_mb:.0%} do tamanho do CSV")


def exercicio_4():
    """Split 70/15/15 com seed fixa e verificação dos tamanhos."""
    dataset = load_dataset("stanfordnlp/imdb", split="train")
    total = len(dataset)

    # Números inteiros evitam diferenças de arredondamento
    n_test = round(total * 0.15)
    n_val = round(total * 0.15)

    # 1ª divisão: separa o teste
    split1 = dataset.train_test_split(test_size=n_test, seed=42)
    test_ds = split1["test"]

    # 2ª divisão: parte do que sobrou e separa a validação
    split2 = split1["train"].train_test_split(test_size=n_val, seed=42)
    train_ds = split2["train"]
    val_ds = split2["test"]

    print(f"Train: {len(train_ds)} ({len(train_ds) / total:.1%})")
    print(f"Val:   {len(val_ds)} ({len(val_ds) / total:.1%})")
    print(f"Test:  {len(test_ds)} ({len(test_ds) / total:.1%})")
    print(f"Soma:  {len(train_ds) + len(val_ds) + len(test_ds)} de {total}")


EXERCICIOS = {
    "1": exercicio_1,
    "2": exercicio_2,
    "3": exercicio_3,
    "4": exercicio_4,
}


if __name__ == "__main__":
    escolhidos = sys.argv[1:] or list(EXERCICIOS)
    for numero in escolhidos:
        print(f"\n===== Exercício {numero} =====")
        EXERCICIOS[numero]()