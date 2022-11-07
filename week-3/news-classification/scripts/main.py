import wandb

from utils import set_env, load_model, load_data, train_test_split, model_embedding, get_embedding_list, distance


def main():
    set_env()
    wandb.login()
    data = load_data()
    train_data, test_data = train_test_split(data)
    labels_list = train_data['target_numerical'].values
    descriptions_list = train_data['text'].values
    tokenizer, model = load_model()
    print(f"tokenizer1 : {tokenizer.model_max_length}")
    print(f"tokenizer2 : {tokenizer.max_model_input_sizes}")
    print(f"tokenizer3 : {tokenizer.pretrained_init_configuration}")
    embeddings_list = get_embedding_list(labels_list, descriptions_list, tokenizer, model)
    test_labels_list = test_data['target_numerical'].values
    test_descriptions_list = test_data['text'].values
    pred_labels = []
    correctly_predicted = 0
    amount_of_prediction = 0
    for ind in range(len(test_descriptions_list)):
        wandb.init(
            # Set the project where this run will be logged
            project="project-test2",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{ind}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "roberta",
                "dataset": "ukrainian news classification",
            }
        )
        label = test_labels_list[ind]
        descr = test_descriptions_list[ind]

        test_emb = model_embedding(descr, tokenizer, model)

        scores = list((distance(el, test_emb), k) for k, el in embeddings_list.items())
        amount_of_prediction += 1
        if scores:
            sorted_scores = sorted(scores, key=lambda x: x[0])
            best_preds = sorted_scores[0][1]
            if label == best_preds:
                correctly_predicted += 1
            accuracy = float(correctly_predicted / amount_of_prediction)
            print(accuracy)
            print(best_preds, label)
            wandb.log({"acc": accuracy})
            wandb.log({"actual/predicted": (label, best_preds)})
            pred_labels.append(best_preds)

    wandb.finish()
    

if __name__ == "__main__":
    main()
