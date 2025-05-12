# # gpt_4o
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction memory

# # qwen_2_5_3b
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_3b --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_3b --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_3b --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction memory

# # qwen_2_5_coder_7b
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_coder_7b --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_coder_7b --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_coder_7b --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction memory

# # qwen_2_5_7b_instruct
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_7b_instruct --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_7b_instruct --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name qwen_2_5_7b_instruct --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction memory

# # llama_4_scout_17b_16e_instruct
# python venus_batch_generation.py --original_dataset_split_name llama_4_scout_17b_16e_instruct --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name llama_4_scout_17b_16e_instruct --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name llama_4_scout_17b_16e_instruct --afterburner_split_name gpt_4o --afterburner_model_name gpt-4o --efficiency_instruction memory

# gpt_4o
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction memory

# # claude_3_5_haiku
# python venus_batch_generation.py --original_dataset_split_name claude_3_5_haiku --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name claude_3_5_haiku --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name claude_3_5_haiku --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction memory


# # claude_3_7_sonnet
# python venus_batch_generation.py --original_dataset_split_name claude_3_7_sonnet --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name claude_3_7_sonnet --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name claude_3_7_sonnet --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction memory

# # deepseek_v3
# python venus_batch_generation.py --original_dataset_split_name deepseek_v3 --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name deepseek_v3 --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name deepseek_v3 --afterburner_split_name claude_3_7_sonnet --afterburner_model_name claude-3-7-sonnet-20250219 --efficiency_instruction memory

# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name o4_mini --afterburner_model_name o4-mini-2025-04-16 --efficiency_instruction integral
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name o4_mini --afterburner_model_name o4-mini-2025-04-16 --efficiency_instruction time
# python venus_batch_generation.py --original_dataset_split_name gpt_4o --afterburner_split_name o4_mini --afterburner_model_name o4-mini-2025-04-16 --efficiency_instruction memory


python venus_batch_generation.py --afterburner_dataset_config qwq_32b --afterburner_model_name Qwen/QwQ-32B --efficiency_instruction integral --original_dataset_config qwq_32b
python venus_batch_generation.py --afterburner_dataset_config qwq_32b --afterburner_model_name Qwen/QwQ-32B --efficiency_instruction time --original_dataset_config qwq_32b
python venus_batch_generation.py --afterburner_dataset_config qwq_32b --afterburner_model_name Qwen/QwQ-32B --efficiency_instruction memory --original_dataset_config qwq_32b

