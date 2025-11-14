import argparse
import asyncio
import uuid

from tqdm import tqdm


async def run_training(model, train_size, message, mode, device, runs_set, config, exp):
    """
    Runs the training script with a specified training size.
    """
    dim = config[0]
    depths = config[1]
    model_size = config[2]  # sra or regular
    sra_ratios = config[3]
    print(
        f"Starting {exp} {model} training with train_size={train_size} {mode}, on {device}, with config {config}"
    )
    process = await asyncio.create_subprocess_exec(
        "python",
        f"train_test_swinunet_sra_{model}.py",
        "--message",
        str(message),
        "--train_size",
        str(train_size),
        "--mode",
        str(mode),
        "--device",
        str(device),
        "--runs_set",
        str(runs_set),
        "--dim",
        str(dim),
        "--depths",
        str(depths),
        "--model_size",
        str(model_size),
        "--sra-ratios",
        str(sra_ratios),
        "--exp",
        str(exp),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f"Training with train_size={train_size} completed successfully.")
    else:
        print(f"Training with train_size={train_size} failed.")
        print("Error:", stderr.decode())

    return process.returncode, device


async def main():
    parser = argparse.ArgumentParser(description="Train/Test driver")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to be trained/tested",
        default="",
    )
    parser.add_argument(
        "--exp",
        "-e",
        help="Type of experiment",
        default="sra",
    )
    parser.add_argument(
        "--devices",
        "-d",
        help="Lists of devices",
        default=None,
    )
    args = parser.parse_args()

    runs_set = uuid.uuid4().hex

    if args.exp == "sra":
        configs = list(
            [
                (96, [2, 2, 2, 2], "sra", [1, 1, 2, 4]),
                (96, [2, 2, 2, 2], "regular", [1, 1, 1, 1]),
            ]
        )
    elif args.exp == "sra_75":
        configs = list(
            [
                (96, [2, 2, 2, 1], "sra_75", [1, 1, 2, 4]),
                (96, [2, 2, 2, 2], "regular", [1, 1, 1, 1]),
            ]
        )
    else:
        configs = list(
            [
                (96, [2, 2, 2, 1], "75", [1, 1, 1, 1]),
                (96, [2, 2, 2, 2], "regular", [1, 1, 1, 1]),
            ]
        )

    train_sizes = [1]  # Example train sizes (fractions of dataset)

    tasks = set()
    max_concurrent = 2  # Number of parallel training processes

    modes = ["pretrained"]
    devices = [
        f"cuda:{i}"
        for i in (range(max_concurrent) if args.devices is None else eval(args.devices))
    ]

    count = 0

    print("runs_set uuid: ", runs_set)

    for config in tqdm(configs, total=len(configs)):
        for size in train_sizes:
            for mode in modes:
                while len(tasks) >= max_concurrent:
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        return_code, device = task.result()
                        devices.append(device)
                        if return_code != 0:
                            print(
                                f"Error on size {size}, mode {mode}, device {device}, config {config}"
                            )

                last_device = devices.pop(0)
                message = f"Training with train_size={size} {mode}, on {last_device}, with config {config}"
                task = asyncio.create_task(
                    run_training(
                        args.model,
                        size,
                        message,
                        mode,
                        last_device,
                        runs_set,
                        config,
                        args.exp,
                    )
                )
                tasks.add(task)
                count += 1

    await asyncio.gather(*tasks)  # Wait for all remaining tasks to complete


if __name__ == "__main__":
    asyncio.run(main())
