import { Command } from 'commander';

export function stopCommand(): Command {
  const command = new Command('stop');
  command.description('stop command');
  return command;
}
