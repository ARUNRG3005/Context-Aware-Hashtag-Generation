import os
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def profile_system():
    print("Initializing Deep System Profiling...")
    
    # 1. Baseline System State
    process = psutil.Process(os.getpid())
    baseline_ram = process.memory_info().rss / (1024 * 1024)
    battery = psutil.sensors_battery()
    baseline_battery = battery.percent if battery else "N/A"
    
    print(f"Baseline RAM: {baseline_ram:.1f} MB")
    if battery:
        print(f"Baseline Battery: {baseline_battery}% (Plugged In: {battery.power_plugged})")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Load Model
    start_load = time.time()
    model_path = os.path.join("d:\\hashtag-generator", "checkpoints", "best_model")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    load_time = time.time() - start_load
    
    loaded_ram = process.memory_info().rss / (1024 * 1024)
    print(f"RAM after loading model: {loaded_ram:.1f} MB (Delta: +{loaded_ram - baseline_ram:.1f} MB)")
    if device.type == "cuda":
        print(f"GPU VRAM Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB")
        
    # 3. Profiling Loop
    print("\nStarting 100x Inference Stress Test...")
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "India's general elections see a massive voter turnout across multiple states.",
        "The new AI language model has surpassed human baseline in coding tasks.",
        "The stock market experienced extreme volatility due to inflation reports.",
        "Scientists have discovered a new exoplanet in the habitable zone."
    ] * 20
    
    # Warmup
    inputs = tokenizer(dummy_texts[:2], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
        
    psutil.cpu_percent() # discard first call
    
    latencies = []
    cpu_spikes = []
    
    for text in dummy_texts:
        start_inf = time.perf_counter()
        
        inputs = tokenizer([text], truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
            
        end_inf = time.perf_counter()
        latencies.append((end_inf - start_inf) * 1000) # ms
        cpu_spikes.append(psutil.cpu_percent())
        
    # 4. Compile Metrics
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_cpu = sum(cpu_spikes) / len(cpu_spikes)
    max_cpu = max(cpu_spikes)
    
    peak_ram = process.memory_info().rss / (1024 * 1024)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0
    
    battery_end = psutil.sensors_battery()
    battery_drain = (baseline_battery - battery_end.percent) if (battery and not battery.power_plugged) else 0
    
    # 5. Generate Report
    report = f"""# AI Hardware Profiling Report

## System State
* **Hardware:** {device.type.upper()}
* **Initial RAM:** {baseline_ram:.1f} MB
* **Initial Battery:** {baseline_battery}%

## Model Footprint
* **Model Load Time:** {load_time:.2f} seconds
* **RAM Allocation (System):** {loaded_ram - baseline_ram:.1f} MB
* **VRAM Allocation (GPU):** {peak_vram:.1f} MB

## Inference Stress Test (100 Iterations)
* **Average Speed:** {avg_latency:.1f} ms per article
* **95th Percentile Speed:** {p95_latency:.1f} ms per article
* **Throughput:** {1000 / avg_latency:.1f} articles per second
* **Average CPU Utilization:** {avg_cpu:.1f}%
* **Peak CPU Utilization:** {max_cpu:.1f}%
* **Peak RAM Usage:** {peak_ram:.1f} MB

## Power Consumption
* **Battery Drain During Test:** {battery_drain}%
* **Current Battery Level:** {battery_end.percent if battery_end else 'N/A'}%
"""
    
    print("\n" + report)
    
    with open("profiling_report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
if __name__ == "__main__":
    profile_system()
