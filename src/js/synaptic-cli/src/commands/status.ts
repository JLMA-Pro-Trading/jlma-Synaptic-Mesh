import { Command } from 'commander';

export function statusCommand(): Command {
  const command = new Command('status');
  command.description('status command');
  return command;
}
