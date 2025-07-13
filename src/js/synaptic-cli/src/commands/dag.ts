import { Command } from 'commander';

export function dagCommand(): Command {
  const command = new Command('dag');
  command.description('dag command');
  return command;
}
