%
% NeuroSky-Lab - NeuroSky data acquisition interface under Matlab
% Copyright (c) 2008-2009 NeuroSky, Inc. All Rights Reserved
%
% version 1.1.3

function EEG = neuroskylab(key, fig, guifig, pan)

if nargout > 0
    EEG = [];
end;
warning('off');
global s sevent;
evalin('base'  , 'global s sevent;');
warning('on');

% Start-up default options
portstatus = 'closed';

opt.firmwarecorrect = 2000;  % Sensor-to-ADC gain correction factor
opt.firmware        = 1.7;   % Firmware version identifier
opt.fid_thinkgear  = [];
opt.fid_streamlog  = [];
opt.portstatus     = portstatus;
opt.command        = 'reset';
opt.hdl_savetext   = [];

bitlevels = [-2048 2047];  % bitlevels[] numbers are used to calculate actual 
                           % voltage at input.  If 8-bits raw values comes in 
                           % bitlevels[] will be changed to 2^8-1 = 255
                           
% Gui Functions
if nargin > 1
    opt = get(fig, 'userdata');    
    
    switch key
 
        % slider
        % ------       
        case 'slide'
 
            obj = findobj(fig, 'tag', 'eegslider');
            val = get(obj, 'value');
            if val == get(obj, 'max')
                opt.command    = 'continue';
                set(obj, 'userdata', 1);
            else
                opt.update  = 'off';
                opt.datapos = val/get(obj, 'max');
                opt.command    = 'plotposition';
                set(obj, 'userdata', 0);
            end;

        % close window
        % ------------
        case 'closegui'
 
            %src is the handle of the object generating the callback (the source of the event)
            %evnt is the The event data structure (can be empty for some callbacks)

            opt = get(fig, 'userdata');
            opt.command = 'quit';
            set(fig, 'userdata', opt);
            
            % close the GUI if it has not been updated for the past 8 seconds
            if ~isfield(opt, 'time'), delete(fig); end;
            try
                tmp = toc;
                if (tmp-opt.time)*10 > 2, evalin('base', [ 'delete(' int2str(fig) ');' ]); end;
                if (tmp-opt.time)*10 > 5, disp('Type "delete(gcf)" to delete current figure'); end;
            catch, evalin('base', [ 'delete(' int2str(fig) ');' ]); end;
            disp('Bye.');
            return;
            
        % key pressed
        % -----------
        case 'keypressed'
 
            opt.event(end+1).type  = get(fig,'Currentkey');
            opt.event(end).latency = toc;            
            opt.event(end).latencystr = datestr(now, 'yyddmmHHMMSSFFF');            
            axes(opt.hdl_axisplot);
            
        % replay streamlog file
        % ---------------------
        case 'replay'
 
            [filename filepath] = uigetfile('*.txt', 'Pick a Streamlog file');
            if filename(1) == 0, return; end;
            
            if ~isempty(opt.fileid_replay), fclose( opt.fileid_replay ); end;
            filename = fullfile(filepath, filename);
            opt.fileid_replay = fopen(filename, 'r');
            %opt.last_saved_file = filename;
            if opt.fileid_replay == -1
                warndlg('Cannot open file', 'Warning', 'modal');
                opt.fileid_replay = [];
            else
                opt.command = 'reset';
            end;

        % spectrum window options
        % -----------------------
        case 'spectrum'
            
            valcheckbox = fastif(strcmpi(opt.fftplottype, '3D'), 1, 0);
            plotopt = { 'Plot as histogram' 'Plot in 2-D curve' 'Plot as 3-D surface' };
            txt_scale    = fastif( opt.fftlog, 'Plotting scale log(uV^2/Hz)', 'Plotting scale (uV^2/Hz)');
            plotoptshort = { 'hist' '2d' '3d' };
            valopt = strmatch(lower(opt.fftplottype), plotoptshort);
            cb_log = [ 'tmpobj = findobj(gcbf, ''tag'', ''fftlog'');' ...
                       'set(findobj(gcbf, ''tag'', ''scaletext''), ''string'',' ...
                       '   fastif( get(tmpobj, ''value''), ''Plotting scale log(uV^2/Hz)'', ''Plotting scale (uV^2/Hz)''));' ...
                       'set(findobj(gcbf, ''tag'', ''fftscale''), ''string'', fastif( get(tmpobj, ''value''), ''-50 10'', ''0 0.03'')); clear tmpobj;'
                       ];
            cb_type = [ 'if get(gcbo, ''value'') == 3,' ...
                        '  set(findobj(gcbf, ''tag'', ''fftlog''), ''value'', 1);' ...
                        '  if str2num(get(findobj(gcbf, ''tag'', ''fftrefreshrate''), ''string'')) < 1000' ...
                        '     set(findobj(gcbf, ''tag'', ''fftrefreshrate''), ''string'', ''1000'');' ...
                        '  end;' ...
                        'end;' cb_log ];
                   
%             Code to create the figure and save it
%             uilist = { { 'Style' 'text' 'string' 'Plotting window length (s)' } ...
%                        { 'Style' 'edit' 'string' num2str(opt.fftwinlen) 'tag' 'fftwinlen' } ...
%                        { 'Style' 'text' 'string' txt_scale              'tag' 'scaletext' } ...
%                        { 'Style' 'edit' 'string' num2str(opt.fftscale)  'tag' 'fftscale' } ...
%                        { 'Style' 'text' 'string' 'Show log of power' } ...
%                        { 'Style' 'checkbox' 'string' '' 'value' opt.fftlog 'callback' cb_log 'tag' 'fftlog' } ...
%                        { 'Style' 'text' 'string' 'Plotting format' } ...
%                        { 'Style' 'popupmenu' 'string' plotopt 'value' valopt 'tag' 'fftplottype' } ...
%                        { 'Style' 'text' 'string' 'Refresh rate (ms)'} ...
%                        { 'Style' 'edit' 'string' num2str(opt.fftrefreshrate/opt.srate*1000,3) 'tag' 'fftrefreshrate' } };
%             inputgui( 'geometry', {[1 1] [1 1] [1 1] [1 1] [1 1] }, 'uilist', uilist, 'mode', 'plot');
%             hgsave(gcf, 'spectrum.fig', '-v6');

            sfig = hgload('spectrum.fig');
            recenter(sfig);
            set(findobj(sfig, 'tag', 'fftwinlen'      ), 'string',   num2str(opt.fftwinlen/opt.srate));
            set(findobj(sfig, 'tag', 'scaletext'      ), 'string',   txt_scale);
            set(findobj(sfig, 'tag', 'fftscale'       ), 'string',   num2str(opt.fftscale));
            set(findobj(sfig, 'tag', 'fftlog'         ), 'value' ,   opt.fftlog, 'callback', cb_log);
            set(findobj(sfig, 'tag', 'fftplottype'    ), 'string',   plotopt, 'value', valopt, 'callback', cb_type);
            set(findobj(sfig, 'tag', 'fftrefreshrate' ), 'string',   num2str(opt.fftrefreshrate*1000,3) );
            set(findobj(sfig, 'tag', 'ok'             ), 'callback', [ 'neuroskylab(''spectrumaccept'', ' num2str(fig) ', gcbf);' ] );
                        
        % spectrum window options
        % -----------------------
        case 'spectrumaccept'
 
            plotoptshort = { 'hist' '2d' '3d' };
            try findobj(guifig); catch return; end;
            res = getguioutput(guifig); close(guifig);
            
            tmprefreshrate = str2double(res.fftrefreshrate)/1000;
            if ~isempty(tmprefreshrate) && tmprefreshrate(1) >= 0.1 && tmprefreshrate(1) <= 10, 
                opt.fftrefreshrate = tmprefreshrate; 
            else warndlg('Warning: invalid format or value for panel refresh rate (100 ms to 10000 ms)');
            end;
            
            opt.fftlog      = res.fftlog;
            opt.fftplottype = plotoptshort(res.fftplottype);
            if res.fftplottype == 3, 
                opt.fftlog = 1;
                if opt.fftrefreshrate < 1, opt.fftrefreshrate = 1; end; 
            end;
            tmpwinlen = round(str2double(res.fftwinlen)*opt.srate);
            if ~isempty(tmpwinlen) && tmpwinlen(1) >= 0.1*opt.srate && tmpwinlen(1) <= 1*opt.srate, 
                 opt.fftwinlen = tmpwinlen(1); 
            else warndlg('Warning: invalid format or value for FFT window length (0.1 s to 1 s)');
            end;
            tmpfftscale       = str2num(res.fftscale);             
            if ~isempty(tmpfftscale) && length(tmpfftscale) == 2 && tmpfftscale(1) < tmpfftscale(2), 
                opt.fftscale = tmpfftscale; 
            else warndlg('Warning: invalid format for spectrum scale');
            end;
            
        % panel menu
        % -----------
        case 'panel'
            
            optval = strmatch(opt.pancontent{pan}, opt.pancontentall);
            
%             uilist = { { 'Style' 'text'    'string' 'What to plot?' } ...
%                        { 'Style' 'popupmenu' 'string' optionplot 'callback' cb_opt 'tag' 'pancontent' 'value' optval } ...
%                        { 'Style' 'text'    'string' 'Frequency limits [Min Max]' 'tag' 'optparam' 'userdata' 'opts' } ...
%                        { 'Style' 'edit'    'string' '' 'tag' 'optparamval' 'userdata' 'opts' } ...
%                        { 'Style' 'text'    'string' 'Plotting window length (s)' } ...
%                        { 'Style' 'edit'    'string' num2str(opt.panwinlen(pan)) 'tag' 'panwinlen' } ...
%                        { 'Style' 'text'    'string' 'Plotting scale' } ...
%                        { 'Style' 'edit'    'string' num2str(opt.panscale{pan}) 'tag' 'panscale' } ...
%                        { 'Style' 'text'    'string' 'Refresh rate (ms)'} ...
%                        { 'Style' 'edit'    'string' sprintf('%3.0f', opt.panrefreshrate(pan)/opt.srate*1000) 'tag' 'panrefreshrate' } };
%             inputgui( 'geometry', {[1 1] [1 1] [1 1] [1 1] [1 1] }, 'uilist', uilist, 'geomvert', [1 1 1 1 1], ...
%                       'eval', cb_opt, 'userdata', { num2str(opt.panfreqlimits{pan}) opt.panfunction{pan} }, 'mode', 'plot');
%             hgsave('panel.fig', '-v6');
            
            sfig = hgload('panel.fig');
            recenter(sfig);
            cb_opt = [ 'neuroskylab(''panelselect'', ' num2str(fig) ', ' num2str(sfig) ');' ];
            set(findobj(sfig, 'tag', 'pancontent'     ), 'value' , optval, 'callback', cb_opt, 'string', opt.pancontentall);
            set(findobj(sfig, 'tag', 'panwinlen'      ), 'string', num2str(opt.panwinlen(pan)/opt.srate) );
            set(findobj(sfig, 'tag', 'panrefreshrate' ), 'string', num2str(opt.panrefreshrate(pan)*1000) );
            set(findobj(sfig, 'tag', 'ok'             ), 'callback', [ 'neuroskylab(''panelaccept'', ' num2str(fig) ', gcbf, ' int2str(pan) ');' ] );
            set(sfig, 'userdata', { num2str(opt.panfreqlimits{pan}) opt.panfunction{pan} pan });
            eval(cb_opt);

        case 'panelselect'
 
            objtmp = findobj(guifig, 'tag', 'pancontent');
            userdat = get(guifig,'userdata');
            set(findobj(guifig, 'userdata', 'opts'), 'visible', 'off');
            set(findobj(guifig, 'tag', 'panscale'), 'string', num2str(opt.panscaleall{userdat{3}}{get(objtmp, 'value')}));            
            if get(objtmp, 'value') == 10,
               set(findobj(guifig, 'userdata', 'opts'), 'visible', 'on');
               set(findobj(guifig, 'tag', 'optparam'   ), 'string', 'Frequency limits [Min Max]');
               set(findobj(guifig, 'tag', 'optparamval'), 'string', userdat{1});
            elseif get(objtmp, 'value') == 11,
               set(findobj(guifig, 'userdata', 'opts'), 'visible', 'on');
               set(findobj(guifig, 'tag', 'optparam'   ), 'string', 'Matlab function name');
               set(findobj(guifig, 'tag', 'optparamval'), 'string', userdat{2});
            end;
                   
        case 'panelaccept'

            try findobj(guifig); catch return; end;
            res = getguioutput(guifig); close(guifig);
            
            tmpwinlen  = round(str2double(res.panwinlen)*opt.srate);
            if ~isempty(tmpwinlen) && tmpwinlen(1) >= 0.1*opt.srate && tmpwinlen(1) <= 30*opt.srate, 
                 opt.panwinlen(pan) = tmpwinlen(1); 
            else warndlg('Warning: invalid format or value for panel window length (0.1 s to 30 s)');
            end;             
            tmpscale = str2num(res.panscale);
            if ~isempty(tmpscale) && length(tmpscale) == 2 && tmpscale(1) < tmpscale(2), 
                opt.panscaleall{pan}{res.pancontent} = tmpscale; 
                opt.panscale{pan}                    = tmpscale;
            else if res.pancontent~= 1, warndlg('Warning: invalid format for Panel scale'); end;
            end;
            tmprefreshrate = str2double(res.panrefreshrate)/1000;
            if ~isempty(tmprefreshrate) && tmprefreshrate(1) >= 0.1 && tmprefreshrate(1) <= 10, 
                opt.panrefreshrate(pan) = tmprefreshrate; 
            else warndlg('Warning: invalid format or value for panel refresh rate (100 ms to 10000 ms)');
            end;
            opt.pancontent{pan} = opt.pancontentall{res.pancontent};
            if strcmpi(opt.pancontent{pan}, 'Custom freq.')
                tmpfreqlimits = str2num(res.optparamval);
                if ~isempty(tmpfreqlimits) && length(tmpfreqlimits) == 2 && tmpfreqlimits(1) < tmpfreqlimits(2), 
                    opt.panfreqlimits{pan} = tmpfreqlimits;
                else warndlg('Warning: invalid format for frequency limits');
                end;
            elseif strcmpi(opt.pancontent{pan}, 'Attention') || strcmpi(opt.pancontent{pan}, 'Meditation')
                if isempty(str2num(res.panscale))
                    opt.panscale{pan} = [0 100];
                end;
            elseif strcmpi(opt.pancontent{pan}, 'Custom func.')
                opt.panfunction{pan} = res.optparamval;
            end;
            plotpanel(opt.panaxis(pan), pan, [], { 'Use menu to' 'select measure' 'to plot' });
            
        % scaling and window length
        % -------------------------
        case 'plotoptions'

%             uilist = { { 'Style' 'text' 'string' 'Plotting window length (s)' } ...
%                        { 'Style' 'edit' 'string' num2str(opt.winlen) 'tag' 'winlen' } ...
%                        { 'Style' 'text' 'string' 'Plotting scale (uV)' } ...
%                        { 'Style' 'edit' 'string' num2str(opt.scale) 'tag' 'scale' } ...
%                        { 'Style' 'text' 'string' 'Refresh rate (ms)' 'foregroundcolor' [0.5 0.5 0.5 ]} ...
%                        { 'Style' 'edit' 'string' num2str(opt.refreshrate/opt.srate*1000,3) 'tag' 'refreshrate' } };
%             inputgui( 'geometry', {[1 1] [1 1] [1 1]}, 'uilist', uilist, 'mode', 'plot');
%             hgsave(gcf, 'plotdataoptions.fig', '-v6');

            sfig = hgload('plotdataoptions.fig');
            recenter(sfig);
            set(findobj(sfig, 'tag', 'winlen'      ), 'string' , num2str(opt.winlen/opt.srate));
            set(findobj(sfig, 'tag', 'scale'       ), 'string' , num2str(opt.scale/opt.firmwarecorrect*2*10^6));
            set(findobj(sfig, 'tag', 'refreshrate' ), 'string' , num2str(opt.refreshrate*1000,3));
            set(findobj(sfig, 'tag', 'ok'          ), 'callback', [ 'neuroskylab(''plotoptionsaccept'', ' num2str(fig) ', gcbf);' ] );
            
        case 'plotoptionsaccept'

            %waitfor( findobj('parent', sfig, 'tag', 'ok'), 'userdata');
            try findobj(guifig); catch return; end;
            res = getguioutput(guifig); close(guifig);
            
            tmpwinlen  = round(str2double(res.winlen)*opt.srate);
            if ~isempty(tmpwinlen) && tmpwinlen(1) >= opt.srate/10 &&  tmpwinlen(1) <= 30*opt.srate, 
                 opt.winlen   = tmpwinlen(1); 
                 opt.winlen_a = opt.winlen/opt.srate*opt.srate_a;
            else warndlg('Warning: invalid format or values for data visualization window length (0.1 s to 30 s)');
            end
            tmpscale = str2double(res.scale)*opt.firmwarecorrect/2/10^6;
            if ~isempty(tmpscale) && tmpscale(1) > 0, 
                opt.scale = tmpscale(1);
            else warndlg('Warning: invalid format for plotting scale');
            end
            tmprefreshrate = str2double(res.refreshrate)/1000;
            if ~isempty(tmprefreshrate) && tmprefreshrate(1) >= 0.1 && tmprefreshrate(1) <= 10, 
                opt.refreshrate = tmprefreshrate;
            else warndlg('Warning: invalid format or value for plotting refresh rate (100 ms to 10000 ms)');
            end
            
            % focues on opt.hd1_axiscale
            %axes( opt.hdl_axisscale );
            % change the scaling
            %plotscale(opt.scale/opt.firmwarecorrect, opt.scalevert/opt.firmwarecorrect, opt.nbchan);
            
        % disconnect from chipset
        % -----------------------
        case 'disconnect'

            if ~isempty(opt.fid_thinkgear)
                neuroskylab('stop', fig);
            end
            
            % change GUI and menu
            cb_connect = 'neuroskylab(''connect'', gcbf);';
            set(findobj( fig, 'tag', 'connect_menu'), 'label', 'Connect', 'callback', cb_connect);
            opt.portstatus = 'closed';
            
        % pop-up window to set recording options
        % --------------------------------------
        case 'recordopt'

            cb_thinkgear = 'if ~get(gcbo, ''value''), set(findobj(gcbf, ''tag'', ''save_stream''   ), ''value'', 1); end;';
            cb_streamlog = 'if ~get(gcbo, ''value''), set(findobj(gcbf, ''tag'', ''save_thinkgear''), ''value'', 1); end;';
            cb_reshowfilenames = [ 'if get(findobj(gcf, ''tag'', ''dateinfilename''), ''value''),' ...
                '   strcomment   = sprintf( ''File names currently set to\n- %s_thinkgear_%s.txt\n- %s_stream_%s.txt'',' ...
                '                  get(findobj(gcf, ''tag'', ''basefilename''), ''string''),' ...
                '                  datestr(now, ''ddmmmyyyy_HHhMM;SS''),' ...
                '                  get(findobj(gcf, ''tag'', ''basefilename''), ''string''),' ...
                '                  datestr(now, ''ddmmmyyyy_HHhMM;SS''));' ...
                '   strcomment(find(strcomment == '';'')) = ''m'';' ...
                'else,' ...
                '   strcomment   = sprintf( ''File names currently set to\n- %s_thinkgear.txt\n- %s_stream.txt'',' ...
                '                  get(findobj(gcf, ''tag'', ''basefilename''), ''string''),' ...
                '                  get(findobj(gcf, ''tag'', ''basefilename''), ''string''));' ...
                'end;' ...
                'set(findobj(gcf, ''tag'', ''comment''), ''string'', strcomment); clear strcomment;' ];
                
            if opt.dateinfilename
                strcomment   = sprintf( 'File names currently set to\n- %s_thinkgear_%s%s%s.txt\n- %s_stream_%s%s%s.txt', ...
                            opt.basefilename, datestr(now, 'ddmmmyyyy_HHhMM'), 'm', datestr(now, 'SS'), ...
                            opt.basefilename, datestr(now, 'ddmmmyyyy_HHhMM'), 'm', datestr(now, 'SS'));
            else
                strcomment   = sprintf( 'File names currently set to\n- %s_thinkgear.txt\n- %s_stream.txt', opt.basefilename, opt.basefilename);
            end
            
%             uilist = { { 'Style' 'text'     'string' 'Save what type of file?', 'fontweight', 'bold' } ...
%                        {} { 'Style' 'checkbox' 'string' 'Thinkgear file format' 'tag' 'save_thinkgear' 'callback' cb_thinkgear 'value' opt.save_thinkgear } ...
%                        {} { 'Style' 'checkbox' 'string' 'Stream file format (required for replay)'    'tag' 'save_stream'    'callback' cb_streamlog 'value' opt.save_stream    } ...
%                        {} ...
%                        { 'Style' 'text' 'string' 'File base name:' } ....
%                        { 'style' 'edit' 'string'  opt.basefilename 'tag' 'basefilename' } ...
%                        {} { 'Style' 'checkbox' 'string' 'Add date and time to filename' 'value' opt.dateinfilename 'tag' 'dateinfilename' } ...
%                        { 'Style' 'text' 'string'  strcomment 'tag' 'comment' } };
%             inputgui( 'geometry', {1 [0.1 1] [0.1 1] 1 [1 1] [0.1 1] 1}, 'uilist', uilist, 'geomvert', [1 1 1 1 1 1 3], 'mode', 'plot');
%             set(gcf, 'name', 'Recording options')            
%             hgsave(gcf, 'recordopt.fig', '-v6');
            
            % decode options
            sfig = hgload('recordopt.fig');
            recenter(sfig);
            set(findobj(sfig, 'tag', 'save_thinkgear' ), 'value' , opt.save_thinkgear, 'callback', cb_thinkgear);
            set(findobj(sfig, 'tag', 'save_stream'    ), 'value' , opt.save_stream   , 'callback', cb_streamlog);
            set(findobj(sfig, 'tag', 'basefilename'   ), 'string', opt.basefilename  , 'callback', cb_reshowfilenames);
            set(findobj(sfig, 'tag', 'dateinfilename' ), 'value' , opt.dateinfilename, 'callback', cb_reshowfilenames);
            set(findobj(sfig, 'tag', 'ok'             ), 'callback', [ 'neuroskylab(''recordoptaccept'', ' num2str(fig) ', gcbf);' ] );
            eval(cb_reshowfilenames);
            
        case 'recordoptaccept'

            %waitfor( findobj('parent', sfig, 'tag', 'ok'), 'userdata');
            try findobj(guifig); catch return; end
            res = getguioutput(guifig); close(guifig);
            
            opt.save_thinkgear = res.save_thinkgear;
            opt.save_stream    = res.save_stream;
            opt.basefilename   = res.basefilename;
            opt.dateinfilename = res.dateinfilename;
            
        % send command to chipset
        % -----------------------
        case 'sendcom'

            % ask for command
            if strcmpi(opt.portstatus, 'closed')
                warndlg('No communication port open', 'Warning', 'modal'); return
            end
            commands = { 'Select command below for Firmware 1.7', ...
            '00000001 | RAW OUTPUT: Set/unset to use 57.6k/9600 baud rate', ...
            '00000010 | RAW OUTPUT: Set/unset to enable/disable raw wave output', ...
            '00000100 | RAW OUTPUT: Set/unset to use 10-bit/8-bit raw wave output', ...
            '00010001 | DIAGNOSTIC OUTPUTS: Set/unset to enable/disable battery output', ...
            '00010010 | DIAGNOSTIC OUTPUTS: Set/unset to enable/disable marker output', ...
            '00020001 | MEASURE OUTPUTS: Set/unset to enable/disable poor quality output', ...
            '00020010 | MEASURE OUTPUTS: Set/unset to enable/disable EEG powers output', ...
            '00030001 | ESENSE OUTPUTS: Set/unset to enable/disable attention output', ...
            '00030010 | ESENSE OUTPUTS: Set/unset to enable/disable meditation output', ...
            '11100001 | ACTION COMMANDS: Set to reset data histories and output buffer', ...
            '11100010 | ACTION COMMANDS: Set to request FW version info output', ...
            '11100100 | ACTION COMMANDS: Set to request FW configuration info output', ...
            '11110001 | ATESTMODE: Set to enable Testmode', ...
            '11111111 | TESTMODE: Set to disable Testmode' };
            cb_selcom = [ 'tmpcomlist = get(gcbf, ''userdata'');' ...
                          'if get(gcbo, ''value'') ~= 1,' ...
                          '   set(findobj(gcbf, ''tag'', ''command''), ''string'', tmpcomlist{get(gcbo, ''value'')}(1:8));' ...
                          'end;' ...
                          'clear tmpcomlist;' ];
%              uilist = { { 'Style' 'text'     'string' 'Command (hex. or binary):' } ...
%                         { 'Style' 'edit'     'string' '00000000' 'tag' 'command' } {} ...
%                         { 'Style' 'popupmenu' 'string' commands 'tag' 'comlist' 'callback' cb_selcom} };
%              inputgui( 'geometry', {[2 1 2.2] [1]}, 'uilist', uilist, 'mode', 'plot', 'userdata', commands);
%              hgsave(gcf, 'sendcom.fig', '-v6');
            
            % decode options
            sfig = hgload('sendcom.fig');
            recenter(sfig);
            set(findobj(sfig, 'tag', 'ok'     ), 'callback', [ 'neuroskylab(''sendcomaccept'', ' num2str(fig) ', gcbf);' ] );
            set(findobj(sfig, 'tag', 'comlist'), 'string', commands, 'callback', cb_selcom );
            set(sfig, 'userdata', commands );
            
        case 'sendcomaccept'
            try findobj(guifig); catch return; end
            res = getguioutput(guifig); close(guifig);
            
            % convert result to decimal and write to port
            if length(res.command) == 2 % hexa
                try, val = hex2dec(res.command); catch, warndlg('Warning: invalid command'); end
                fprintf('Sending command %d to headset chip\n', val);
                try, fwrite(s,val,'uint8'); catch, warndlg('Warning: sending command failed'); end
            elseif length(res.command) == 8
                try, val = bin2dec(res.command); catch, warndlg('Warning: invalid command'); end
                fprintf('Sending command %d to headset chip\n', val);
                try, fwrite(s,val,'uint8'); catch, warndlg('Warning: sending command failed'); end
            else
                warndlg('Command must be either 2 characters (hexa) or 8 (binary)', 'Warning', 'modal'); return
            end
            
        % disconnect from chipset
        % -----------------------
        case 'connectother'

            if ~isempty(sevent)
                disp('Serial port opened; closing it');
                fclose(sevent);
            end;
            
            % pop-up window to detect communication port
%             for i=1:64
%                 textport{i} = [ 'COM' int2str(i) ];
%             end;
%             uilist = { { 'Style' 'text' 'string' 'Connection speed (baud)' } ...
%                        { 'Style' 'edit' 'string' '9600' 'tag' 'speed' } ...
%                        { 'Style' 'text' 'string' 'Select port'  } ...
%                        { 'style' 'popupmenu'  'string' textport 'tag' 'port'  } ...
%                        { 'Style' 'text' 'string' 'Other ''key'', val options (parity...):' } ...
%                        { 'Style' 'edit' 'string' '''Parity'', ''even''' 'tag' 'options' } };
%             inputgui( 'geometry', {[2 1] [2 1] 1 1}, 'uilist', uilist, 'mode', 'plot');
%             hgsave(gcf, 'eventport.fig', '-v6');
            
            sfig = hgload('eventport.fig');
            recenter(sfig);
            waitfor( findobj('parent', sfig, 'tag', 'ok'), 'userdata');
            try findobj(sfig); catch return; end
            res = getguioutput(sfig); close(sfig);
            
            % test and open port for events
            if ~isempty(res)
                sevent = serial(textport{res.port});
                try
                    fopen(sevent)
                catch 
                    warndlg('Could not open communication port', 'Warning', 'modal'); sevent = []; return;
                end
                try
                    set(sevent, 'baud', str2double(res.speed));
                    eval( [ 'set(sevent,' res.options ');' ]);
                catch 
                    warndlg('Port parameter error', 'Warning', 'modal');
                    fclose(sevent); return;
                end
                A = fread(sevent,1);
                if isempty(A)
                    warndlg('Port does not deliver data', 'Warning', 'modal');
                    fclose(sevent);
                    sevent = [];
                end
            end
 
        % connect to chipset
        % ------------------
        case 'connect'

            % pop-up window to detect communication port
%             textport = { 'Auto-detect (long delay)' };
%             for i=1:64
%                 textport{i+1} = [ 'COM' int2str(i) ];
%             end;
%             uilist = { { 'Style' 'text' 'string' 'Select board' 'fontweight' 'bold'} ...
%                        { 'Style' 'popupmenu' 'string' { 'MindSet' 'MindSet Pro' 'MindBuilder' } 'tag' 'firmware' } {} ...
%                        { 'Style' 'text' 'string' 'Sampling rate' 'fontweight' 'bold'} ...
%                        { 'Style' 'popupmenu' 'string' { '128 Hz' '256 Hz' } 'tag' 'srate' } {} ...
%                        { 'Style' 'text' 'string' 'Connection speed' 'fontweight' 'bold'} ...
%                        { 'Style' 'popupmenu' 'string' { '9600 bauds' '57600 bauds' } 'tag' 'speed' } ...
%                        {} ...
%                        { 'Style' 'text' 'string' 'Select port' 'fontweight' 'bold' } ...
%                        { 'style' 'listbox'  'string' textport 'tag' 'port' } };
%             inputgui( 'geometry', {1 1 1 1 1 1 1 1 1 1 1}, 'uilist', uilist, 'geomvert', [1 1 0.5 1 1 0.5 1 1 0.5 1 7], 'mode', 'plot');
%             set(gcf, 'name', 'Connect');
%             hgsave(gcf, 'connect.fig', '-v6');
            
            % delete any open COM ports so this can reopen without
            % restarting Matlab
            delete(instrfindall);

            set(fig, 'visible', 'off');
            sfig = hgload('connect_simple.fig');
            recenter(sfig);
            
            %set(findobj(sfig, 'tag', 'firmware'), 'value', 1);
            waitfor( findobj('parent', sfig, 'tag', 'ok'), 'userdata');
            set(fig, 'visible', 'on');
            try findobj(sfig); catch return; end;
            res = getguioutput(sfig); close(sfig);
            drawnow;
            wdlg = warndlg('Connecting (this may take several seconds) ...');
            
%            fprintf ('res.listbox:\n');
%            res.tester
            
            % Autodetect port
            if ~isfield(res, 'speed'),    res.speed    = 2; end
            if ~isfield(res, 'firmware'), res.firmware = 5; end
            if res.speed == 1,      opt.connectspeed = '9600';
            elseif res.speed == 2,  opt.connectspeed = '57600';
            else                    opt.connectspeed = 'auto';
            end
            
            % firmware version select
            opt.srate = 128;
            if res.firmware == 1   
                    opt.firmware = 1.79;
                    opt.firmwarecorrect = 2520;
            elseif res.firmware == 2   
                    opt.firmware = 1.7;
                    opt.firmwarecorrect = 2520;
            elseif res.firmware == 3
                    opt.firmware = 1.5;
                    opt.firmwarecorrect = 4979;
            elseif res.firmware == 5
                    opt.firmware = 1.79;
                    opt.srate = 512;
                    opt.firmwarecorrect = 2000;
            end
            % shenghong
            if res.port == 1
                portind   = 1;
                portfound = 0;
                while ~portfound && portind < 64
                    status = openport(portind, opt.connectspeed, opt.firmware);
                    if status == true, portfound = 1; end
                    portind = portind+1;
                end
            else
                portind = res.port-1;
                status = openport(portind, opt.connectspeed, opt.firmware);
            end;
            if status ~= true
                warndlg('Could not open communication port', 'warning', 'modal');
            else
                opt.portstatus = 'open';
                cb_disconnect = 'neuroskylab(''disconnect'', gcbf);'; % change GUI and menu
                set(findobj( fig, 'tag', 'connect_menu'), 'label', 'Disconnect', 'callback', cb_disconnect);
            end;
            opt.command = 'reset';
            try, delete(wdlg); catch end;
            
        % quit the interface
        % ------------------
        case 'quit'

            opt.command = 'quit';
            
        % quit and transfer the last saved dataset to EEGLAB
        % --------------------------------------------------
        case 'quittransfer'

            opt.command = 'transfer';
            
        % start recording data files
        % --------------------------
        case 'save'

            % check if communication port is opened
            if ~isempty(s)
                status = get(s, 'status');
            end
            if ~strcmpi(status, 'open')
                warndlg('Open communication port first');
                return;
            end
                
            % --- Switch to new file after saving 64 points ---
            opt.line_count = 0;      % 初始化計數器
            opt.file_seq = 1;        % 初始化檔案序號
            % -------------------------------------------------

            % open thinkgear output file
            filename_t = '';
            if opt.save_thinkgear

                filename_t = sprintf('%s_part%d_thinkgear.txt', opt.basefilename, opt.file_seq);
                %if opt.dateinfilename
                %    filename_t = sprintf('%s_thinkgear_%s%s%s.txt', opt.basefilename, ...
                %                datestr(now, 'ddmmmyyyy_HHhMM'), 'm', datestr(now, 'SS'));
                %else
                %    filename_t = sprintf('%s_thinkgear.txt', opt.basefilename);
                %end;
                opt.fid_thinkgear = fopen(filename_t, 'w');
                if opt.fid_thinkgear == -1
                    warndlg('Cannot open file for writing', 'Warning', 'modal'); return;
                end
            end
            
            % open stream output file
            if opt.save_stream
                
                % --- 強制使用 part1 檔名 ---
                filename_s = sprintf('%s_part%d_stream.txt', opt.basefilename, opt.file_seq);

                %if opt.dateinfilename
                %    filename_s = sprintf('%s_stream_%s%s%s.txt', opt.basefilename, ...
                %                    datestr(now, 'ddmmmyyyy_HHhMM'), 'm', datestr(now,'SS'));
                %else
                %    filename_s = sprintf('%s_stream.txt', opt.basefilename);
                %end;
                % --------------------------------

                opt.fid_streamlog = fopen(filename_s, 'w');
                if opt.fid_streamlog == -1
                    if ~isempty(opt.fid_streamlog), fclose(opt.fid_streamlog); end;
                    warndlg('Cannot open file for writing', 'Warning', 'modal'); return;
                end;
            end;

            opt.last_saved_file = filename_t;
                
            % change GUI and menu
            cb_stop = 'neuroskylab(''stop'', gcbf);';
            set(findobj( fig, 'tag', 'save_stop_button'), 'string', 'Stop recording', 'callback', cb_stop);
            set(findobj( fig, 'tag', 'save_stop_menu')  , 'label' , 'Stop recording', 'callback', cb_stop);
            axes(opt.hdl_axisplot); hold on;
            opt.hdl_savetext(1) = text(0.15,0.8, 'Saving', 'unit', 'normalized');
            opt.hdl_savetext(2) = text(0.25,0.3, 'Data...', 'unit', 'normalized');
            set(opt.hdl_savetext, 'fontsize', 30, 'fontweight', 'bold', 'color', [.8 .8 .8]);
            
        % strop recording data file
        % -------------------------
        case 'stop'

            % save events if necessary
            if ~isempty(opt.fid_thinkgear)
                if ~isempty(opt.event)
                    if isempty(opt.last_saved_file)
                        disp('Warning: events not saved (only saved in conjunction with a thinkgear file)');
                    else
                        ind = findstr('.txt', opt.last_saved_file);
                        filename = [ opt.last_saved_file(1:ind-1) '_events.txt' ];
                        fid = fopen(filename, 'w');
                        if fid == -1, disp('Cannot open event file'); 
                        else
                            for i=1:length(opt.event)
                                lat = opt.event(i).latencystr;
                                % fprintf(fid, '%.10d.%.3d: %s\n', floor(lat), floor(rem(lat,1)*1000), opt.event(i).type);
                                fprintf(fid, '%s: %s\n', lat, opt.event(i).type);
                            end;
                            fclose(fid);
                        end;
                    end;
                end;
            end;
            
            try delete(opt.hdl_savetext); catch end;
            opt.hdl_savetext = [];
            opt.command = 'stoprecording';
            
            % change GUI and menu
            cb_save = 'neuroskylab(''save'', gcbf);';
            set(findobj( fig, 'tag', 'save_stop_button'), 'string', 'Start recording', 'callback', cb_save);
            set(findobj( fig, 'tag', 'save_stop_menu')  , 'label' , 'Start recording', 'callback', cb_save);
    end;
    
    set(fig, 'userdata', opt);
    return;
end;

clear functions; % reset persistent variables

% test if port already present
% ----------------------------
if exist('s','var') == 1;
    try
        if strcmpi(get(s, 'status'), 'open')
            fprintf('Using existing connection %s\n', get(s, 'name'));
            opt.portstatus = 'open';
        end;
    catch
    end;
    disp('If you experience connection problems, restart Matlab');
end;

% detect function to filter stream of bytes
% -----------------------------------------
if exist('filterbitstream')
    disp('Filterbitstream function detected');
    opt.filterbitstream = @filterbitstream;
else
    opt.filterbitstream = [];
end;

% create graphical interface (pull down menus in main window)
% -----------------------------------------------------------
cb_keypressed = 'neuroskylab(''keypressed'', gcbf);';
fig = figure(   'name',  'NeuroSky Lab', ... 
    'numbertitle'    , 'off', ...
    'color'          , [1 1 1] , ...
    'Tag'            ,'Neurosky' , ...
    'menubar'        , 'none', ...
    'keypressfcn'    , cb_keypressed, ...
    'CloseRequestFcn', 'neuroskylab(''closegui'', gcbf);');

%   'resize'         , 'off'  , ...
%    'nextplot', 'new', ...
%   'dockcontrol'    , 'on'  ,

% make menus
% ----------
cb_quit         = 'neuroskylab(''quit''        , gcbf);';
cb_quittransfer = 'neuroskylab(''quittransfer'', gcbf);';
cb_connect      = 'neuroskylab(''connect''     , gcbf);';
cb_connectother = 'neuroskylab(''connectother'', gcbf);';
cb_sendcom      = 'neuroskylab(''sendcom''     , gcbf);';
cb_opt          = 'neuroskylab(''recordopt''   , gcbf);';
cb_save         = 'neuroskylab(''save''        , gcbf);';
cb_replay       = 'neuroskylab(''replay''      , gcbf);';
cb_data         = 'neuroskylab(''plotoptions'' , gcbf);';
cb_spectrum     = 'neuroskylab(''spectrum''    , gcbf);';
cb_upperpanel   = 'neuroskylab(''panel''       , gcbf, [], 1);';
cb_lowerpanel   = 'neuroskylab(''panel''       , gcbf, [], 2);';

% draw 'Start recording' button
h = uicontrol('style', 'pushbutton', 'string', 'Start recording', 'callback', cb_save, ...
              'tag', 'save_stop_button', 'unit', 'normalized', 'position', [0.63 0.93 0.20 0.05]);
try set(h, 'keypressfcn' , cb_keypressed); catch end; % not compatible with Matlab 6.5
m_f = uimenu(gcf, 'Label', 'File');
p_f = uimenu(gcf, 'Label', 'Panel');
uimenu(m_f, 'Label', 'Connect chipset'              , 'callback', cb_connect, 'tag', 'connect_menu');
uimenu(m_f, 'Label', 'Send command to chipset'      , 'callback', cb_sendcom);
%uimenu(m_f, 'Label', 'Connect other computer'       , 'callback', cb_connectother, 'enable', 'off');
uimenu(m_f, 'Label', 'Recording options'            , 'callback', cb_opt, 'separator', 'on');
uimenu(m_f, 'Label', 'Start recording'              , 'callback', cb_save, 'tag', 'save_stop_menu');
uimenu(m_f, 'Label', 'Replay data file'             , 'callback', cb_replay);
uimenu(m_f, 'Label', 'Export saved dataset to EEGLAB and quit', 'callback', cb_quittransfer, 'separator', 'on');
uimenu(m_f, 'Label', 'Quit'                         , 'callback', cb_quit);
uimenu(p_f, 'Label', 'Data acquisiton panel'        , 'callback', cb_data);
uimenu(p_f, 'Label', 'Spectrum panel'               , 'callback', cb_spectrum);
uimenu(p_f, 'Label', 'Upper right panel'            , 'callback', cb_upperpanel);
uimenu(p_f, 'Label', 'Lower right panel'            , 'callback', cb_lowerpanel);

% build userdata array
% --------------------
% subplot(2,2,3); hist(rand(1,100)*64, 20); title('Data spectrum');

% upper right panel space holder
opt.panaxis(1) = axes('unit', 'normalized', 'position', [0.55 0.54 0.35 0.32]); 
set(gca, 'xticklabel', [], 'yticklabel', []); box on;
% lower right panel space holder
opt.panaxis(2) = axes('unit', 'normalized', 'position', [0.55 0.1 0.35 0.34]); 
set(gca, 'xticklabel', [], 'yticklabel', []); box on;

% write 'EEG data' label
axes('unit', 'normalized', 'position', [0.10 0.58 0.33 0.333]); axis off;
title('EEG data');
% EEG data graph box
opt.hdl_axisfft    = axes('unit', 'normalized', 'position', [0.13 0.10 0.33 0.34]); set(gca, 'xticklabel', [], 'yticklabel', []); box on;
% Spectrum graph box
opt.hdl_axisplot   = axes('unit', 'normalized', 'position', [0.13 0.58 0.33 0.32]); set(gca, 'xticklabel', [], 'yticklabel', []); box on; 
% pesudo graph box used to draw scale figure ( |---| uV )next to eeg data (unused)
% opt.hdl_axisscale  = axes('unit', 'normalized', 'position', [0.44 0.58 0.03 0.34]); set(gca, 'xticklabel', [], 'yticklabel', []); axis off;

% draw slider
slider = uicontrol('Parent',gcf, ...
        'Units', 'normalized', ...
        'Position', [0.13 0.50 0.33 0.03], ...
        'Style','slider', ...
        'sliderstep', [0.1 0.2], ...
        'min', 0, 'max', 1, ...
        'Tag','eegslider', ...
        'userdata', 1, ...
        'callback', 'neuroskylab(''slide'', gcbf);', 'value', 0);
try 
    set(slider,          'keypressfcn' , cb_keypressed); 
    set(opt.hdl_axisfft, 'keypressfcn' , cb_keypressed);
catch
end; % not compatible with Matlab 6.5

opt.connectspeed   = '9600';
opt.srate          = 512;
opt.srate_a        = 32;
opt.fileid_replay  = [];
opt.event          = [];

% recording options
opt.basefilename   = 'Log';
opt.save_thinkgear = 1;
opt.save_stream    = 1;
opt.dateinfilename = 1;

% scroll data options
opt.nbchan         = 0;     % number of channels eeg
opt.nbchan_a       = 0;     % number of channels accellerometers
opt.refreshrate    = 0.2;   
opt.winlen         = 1024;      % window length
opt.winlen_a       = 256;      % window length
opt.scale          = 1;        % scale in microvolt
opt.scalevert      = 2;

% fft options
opt.fftwinlen      = 512;      % window length for FFT (1 second)
opt.fftscale       = [0 0.3]; % scale in microvolt
opt.fftplottype    = 'hist'; % default display type. can be hist, 3d, or 2d
opt.fftrefreshrate = 0.5;
opt.fftlog         = 0;

% panel options
opt.panwinlen      = [ 4096 4096 ];  %length of each window
opt.pancontentall  = { 'None' 'Attention' 'Meditation' 'EEG delta' 'EEG theta' 'EEG alpha1' ...
                           'EEG alpha2' 'EEG beta1' 'EEG beta2' 'Custom freq.' 'Custom func.' };
opt.panscaleall    = { { [] [0 100] [0 100] [0 500000] [0 100000] [0 50000] [0 50000] [0 40000] [0 40000] [-80 -50] [-0.0003 0.0003] } ...
                       { [] [0 100] [0 100] [0 500000] [0 100000] [0 50000] [0 50000] [0 40000] [0 40000] [-80 -50] [-0.0003 0.0003] } };
opt.panscale       = { [0 100] [0 100] };
opt.panrefreshrate = [ 1 1 ];
opt.panfreqlimits  = { [8 12] [8 12] };
opt.panfunction    = { 'myfilter' 'myfilter' };
%opt.pancontent     = { 'Custom freq.' 'Meditation' }; 
opt.pancontent     = { 'Attention' 'Meditation' }; 
%opt.pancontent     = { 'None' 'None' }; 

% get previsouly stored parameters
if nargin < 1
    try,
        tmpstatus = opt.portstatus;
        tmpaxis1  = opt.panaxis;
        tmpaxis2  = opt.hdl_axisplot;
        tmpaxis3  = opt.hdl_axisfft;
        load('-mat', 'options.mat');
        opt.fileid_replay = [];
        opt.panaxis      = tmpaxis1;
        opt.hdl_axisplot = tmpaxis2;
        opt.hdl_axisfft  = tmpaxis3;
        opt.portstatus   = tmpstatus;
        opt.event        = [];
        opt.pause        = 0;
        opt.command      = 'reset';
    catch, end;
end;

opt.datapos        = 0;
set(fig, 'userdata', opt);

% try connecting 
% --------------
if strcmpi(opt.portstatus, 'closed');
    neuroskylab('connect', fig);
else
    cb_disconnect = 'neuroskylab(''disconnect'', gcbf);';
    set(findobj( fig, 'tag', 'connect_menu'), 'label', 'Disconnect', 'callback', cb_disconnect);
end;

% flush buffers
% -------------
opt = get(fig, 'userdata');
if strcmpi( opt.portstatus, 'open'), fscanf(s); end;
if ~isempty(sevent), fscanf(sevent); end;
pause(0.1);

% loop until stop button is pressed
% ---------------------------------
a = now;
plotDelay      = 0;
plotDelayFFT   = 0;
storeDelayFFT  = 0;
plotDelaySlider = 0;
plotDelaypanel = [0 0];
freqs          = [];
lastdatapos    = 0;
count_tmp      = 0;
estimatesrate  = [];
sliderrefreshrate = 0.5;

% while strcmpi(opt.command, 'no')
%     opt = get(fig, 'userdata');
%     [payloadBuffer rawchecksum ] = readonepacket(0);
%     plotDelay = plotDelay+1;
%     if plotDelay == 128
%         plotDelay = 0;
%         toc; tic
%     end;
% end;
% return;

tic;
while ~strcmpi(opt.command, 'quit') && ~strcmpi(opt.command, 'transfer')
    
    opt = get(fig, 'userdata');
    opt.time = toc;
    set(fig, 'userdata', opt);
    
    % reset values
    % ------------
    if strcmpi(opt.command, 'reset')
        % replot scale
        % ----------
        maxpnts = 3600*2; % one hour at 128 Hz
        % axes( opt.hdl_axisscale );

        % plotscale not needed now because we display absolute voltages at
        % sensor
        %        plotscale(opt.scale/opt.firmwarecorrect, opt.scalevert/opt.firmwarecorrect, opt.nbchan+opt.nbchan_a);

        data.values      = zeros(opt.nbchan  , maxpnts*opt.srate); %nskdata(opt.srate, opt.nbchan, maxpnts);
        data.time        = zeros(1           , maxpnts*opt.srate); %nskdata(opt.srate, opt.nbchan, maxpnts);
        data.count       = 1;
        data.srate       = opt.srate;
        data.nbchan      = opt.nbchan;

        data2.values     = zeros(opt.nbchan_a, maxpnts); %nskdata(opt.srate, opt.nbchan, maxpnts);
        data2.count      = 1;
        data2.srate      = opt.srate_a;
        data2.nbchan     = opt.nbchan_a;

        % initializing arrays
        % -------------------
        battery.values    = zeros(1, maxpnts);
        battery.count     = 1;
        battery.srate     = 1; % 1Hz

        signalqual.values = zeros(1, maxpnts);
        signalqual.count  = 1;
        signalqual.srate  = 1;

        attention.values  = zeros(1, maxpnts);
        attention.count   = 1;
        attention.srate   = 1;

        meditation.values = zeros(1, maxpnts);
        meditation.count  = 1;
        meditation.srate  = 1;

        marker.values     = zeros(1, maxpnts);
        marker.count      = 1;
        marker.srate      = 1;

        offhead.values    = zeros(1, maxpnts);
        offhead.count     = 1;
        offhead.srate     = 1;

        flat.values       = zeros(1, maxpnts);
        flat.count        = 1;
        flat.srate        = 1;

        excess.values     = zeros(1, maxpnts);
        excess.count      = 1;
        excess.srate      = 1;

        lowpower.values   = zeros(1, maxpnts);
        lowpower.count    = 1;
        lowpower.srate    = 1;

        highpower.values  = zeros(1, maxpnts);
        highpower.count   = 1;
        highpower.srate   = 1;

        errorflags.values = zeros(1, maxpnts);
        errorflags.count  = 1;
        errorflags.srate  = 1;

        power.values      = zeros(8, maxpnts); % power values come only once a second
        power.count       = 1;                         % uses 2 becuase power.values is an array
        power.srate       = 1;
        % note: power varialbe is used in fft in calculation of power
        % spectrum

        eventchan.values  = zeros(1, maxpnts);

        ffthist.values    = zeros(maxpnts, 64)*NaN;
        ffthist.count     = 1;
        ffthist.srate     = opt.fftwinlen/opt.srate;

        opt.command = 'continue';
        set(fig, 'userdata', opt);

    end;

    % stop recording
    % --------------
    if strcmpi(opt.command, 'stoprecording')
        % close files
        if ~isempty(opt.fid_thinkgear), fclose(opt.fid_thinkgear); opt.fid_thinkgear = []; end;
        if ~isempty(opt.fid_streamlog), fclose(opt.fid_streamlog); opt.fid_streamlog = []; end;
        opt.command = 'continue';
        set(fig, 'userdata', opt);
    end;

    fid1 = opt.fid_thinkgear;
    fid2 = opt.fid_streamlog;

    % read event channel
    % ------------------
    if ~isempty(sevent)
        eventchan( event.count) = fread(sevent, 1);
    end;

    if strcmpi(opt.portstatus, 'closed') && ~isempty(s)
        try, fclose(s); catch, end;
    end;

    % if port is closed and fileid_replay is empty
    if ~strcmpi(opt.portstatus, 'open') && isempty(opt.fileid_replay) 
        pause(0.05);
        %plotdata(opt.hdl_axisplot, []);
    else % port is open so get data from com port

        % read packets, can be from either serial port of file
        [allpayloadBuffer rawbytes waitflag ] = read_packets(opt.fileid_replay, opt.filterbitstream);
        if isempty(allpayloadBuffer)
            [allpayloadBuffer rawbytes waitflag ] = read_packets(opt.fileid_replay, opt.filterbitstream);
        end;

        % close the file when all of it is read.
        if ~isempty(opt.fileid_replay) && feof(opt.fileid_replay), 
            fclose(opt.fileid_replay); opt.fileid_replay = []; 
            set(fig, 'userdata', opt); 
            disp('End of file reached');
        end;

        % write streamlog and thinkgear file
        % ----------------------------------
        if ~isempty(fid2) % fid2 is streamlog file (unprocessed packet data)
            fprintf(fid2, '%.10d.%.3d: ', floor(toc), floor(rem(toc,1)*1000));
            fprintf(fid2, '%.2X', rawbytes(1));
            for index = 2:length(rawbytes)
                fprintf(fid2, ' %.2X', rawbytes(index));
            end;
            fprintf(fid2, '\n');
        end;

        % sampling rate auto detection
        % ----------------------------
        if isempty(estimatesrate)
            if ~exist('tmpsratetime')
                tmpsratetime = [];
                tmpsratecount = [];
            end;
            tmpsratetime  = [ tmpsratetime toc ];
            tmpsratecount = [ tmpsratecount length(allpayloadBuffer) ];

            if length(tmpsratetime) == 10
                tmpsrate = (tmpsratetime(6:end)-tmpsratetime(5:end-1))./tmpsratecount(6:end);
                tmpsrate = 1./tmpsrate;
                estimatesrate = round(mean(tmpsrate));
                classicalsr = [128 512 ]; % classical sampling rates
                [ tmp ind ] = min(abs(estimatesrate - classicalsr));
                estimatesrate  = classicalsr(ind);
                %fprintf('Sampling rate is %3.2f Hz, bitlevel is %d, number of channel is %d\n', this.srate, this.bitlevels, this.nbchan);
                fprintf('Sampling rate is %3.2f Hz\n', estimatesrate);

                % set bitlevel
                if estimatesrate == 128
                     bitlevels = [0 1024]; % Mindset-pro
                     opt.firmwarecorrect = 2520;
                     opt.srate = 128;
                     opt.winlen = 256;
                     opt.fftwinlen = 128;
                     opt.command = 'reset';
                end;
                %else
                %     bitlevels = [-2048 2047]; % ASIC
                %     opt.firmwarecorrect = 2000;
                %     opt.srate = 512;
                %     opt.winlen = 2048;
                %     opt.fftwinlen = 512;
                %     opt.command = 'reset';
                %end;
                set(fig, 'userdata', opt);
            end;
        end;

        % read packets
        % ------------
        for ip = 1:length(allpayloadBuffer) % scan through all the cells
                                            % where each cell should have a
                                            % code, length(optional), and
                                            % value.
            payloadBuffer = allpayloadBuffer{ip}; % put each cell into here

            % string output macro used later to output time and payload
            % strout = sprintf('%.10d.%.3d: [%.2X] ', floor(toc), floor(rem(toc,1)*1000), payloadBuffer(1));
            if payloadBuffer(1) > 127
                strout = sprintf('%s: %.2X %.2X', datestr(now, 'yyddmmHHMMSSFFF'), payloadBuffer(1), length(payloadBuffer)-2);
            else
                strout = sprintf('%s: %.2X %.2X', datestr(now, 'yyddmmHHMMSSFFF'), payloadBuffer(1), length(payloadBuffer)-1);
            end;

            % if payloadBuffer(1) == 144, opt.srate = 256; end;

            switch payloadBuffer(1)

                case 1
                    %read from battery
                    battery.values(battery.count) = payloadBuffer(2);
                    %strout = sprintf([strout '%2.6f'], payloadBuffer(2)/127);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    battery.count  = battery.count  + 1;

                case 2
                    %signalqual
                    signalqual.values(signalqual.count) = payloadBuffer(2);
                    %strout = sprintf([strout '%d'], payloadBuffer(2));
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    signalqual.count   = signalqual.count   + 1;

                case 4
                    %attention
                    %disp('Attention');
                    attention.values(attention.count) = payloadBuffer(2);
                    %strout = sprintf([strout '%d'], payloadBuffer(2));
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    attention.count = attention.count + 1;

                case 5
                    %meditation
                    meditation.values(meditation.count) = payloadBuffer(2);
                    %strout = sprintf([strout '%d'], payloadBuffer(2));
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    meditation.count = meditation.count + 1;

                case 6
                    %raw wave value 8 bit
                    data.values(data.count) = payloadBuffer(2);
                    bitlevels = [0 256];
                    if ~isempty(fid1), 
                        strout = sprintf([strout ' %.2X %1.6f'], data.values(data.count), ...
                            3.3*(data.values(data.count)-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/opt.firmwarecorrect);
                           % (data.values(data.count)/bitlevels*3.3-1.65)/opt.firmwarecorrect);
                    end;
                    data.time(data.count) = toc;
                    data.count = data.count + 1;

                    if opt.nbchan == 0
                        opt.nbchan = 1;
                    end

                case 7
                    %marker
                    marker.values(marker.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    marker.count   = marker.count   + 1;

                case 8 
                    % Off head value (0-128)
                    offhead.values(offhead.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    offhead.count = offhead.count + 1;

                case 9 
                    % Flat value (0-255)
                    flat.values(flat.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    flat.count = flat.count + 1;

                case 10 % 0Ah 
                    % Excess value (0-255)
                    excess.values(excess.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    excess.count = excess.count + 1;

                case 11  % 0Bh
                    % Low Power value (0-100)
                    lowpower.values(lowpower.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    lowpower.count = lowpower.count + 1;

                case 12 % 0Ch
                    % High Power value (0-100)
                    highpower.values(highpower.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    highpower.count = highpower.count + 1;

                case 13 % 0Dh
                    % Error Flags
                    errorflags.values(errorflags.count) = payloadBuffer(2);
                    strout = sprintf([strout ' %.2X'], payloadBuffer(2));
                    errorflags.count = errorflags.count + 1;

                case 128
                    %raw wave value 10-16 bit
                    %data.values = adddata(data.values, payloadBuffer(3)*256+payloadBuffer(4)); %read the next two bytes after value length and combine the two bytes into one value
                    if opt.nbchan == 0
                        opt.nbchan = 1;
                        opt.command = 'reset';
                        set(fig, 'userdata', opt);
                    else
                        %data.values(data.count) = ((payloadBuffer(3)*256+payloadBuffer(4))/1023*3-1.5)/opt.firmwarecorrect; %read the next two bytes after value length and combine the two bytes into one value
                        data.values(data.count) = payloadBuffer(3)*256+payloadBuffer(4); %read the next two bytes after value length and combine the two bytes into one value
                        if data.values(data.count) > 32767
                            data.values(data.count) = data.values(data.count)-65536;
                        end
                        if ~isempty(fid1), 
                            strout = sprintf([strout ' %.2X %.2X %1.6f'], payloadBuffer(3), ...
                            payloadBuffer(4),  3.3*(data.values(data.count)-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/opt.firmwarecorrect);
                        end;
                        data.time(data.count) = toc;
                        data.count = data.count + 1;
                    end;

                case 129
                    %EEG powers
                    count = 1;
                    %fprintf('Power %d\n', length(payloadBuffer));
                    strout2 = '';
                    for i = 3:4:length(payloadBuffer)-3
                        hexstr = sprintf('%.2X%.2X%.2X%.2X', payloadBuffer(i+3), payloadBuffer(i+2), ...
                            payloadBuffer(i+1), payloadBuffer(i));
                        power.values(count, power.count) = hex2float(hexstr); % convert to float
                        strout  = sprintf([ strout ' %.2X %.2X %.2X %.2X' ], payloadBuffer(i+3), payloadBuffer(i+2), ...
                            payloadBuffer(i+1), payloadBuffer(i));
                        strout2 = sprintf([strout2 ' %2.6f '], power.values(count, power.count));
                        %fprintf('%s\n', strout);
                        count = count+1;
                    end;
                    strout = [ strout strout2 ];
                    power.count = power.count + 1;

                case 131 
                    % ASIC EEG Power
                    count = 1;
                    strout2 = '';
                    for i = 3:3:length(payloadBuffer)-2
                        power.values(count, power.count) = payloadBuffer(i)*65536 +payloadBuffer(i+1)*256+ payloadBuffer(i+2);
                        strout = sprintf([strout,' %.2X %.2X %.2X'], payloadBuffer(i), payloadBuffer(i+1), payloadBuffer(i+2));
                        strout2 = sprintf([strout2 ' %2.6f '], power.values(count, power.count));
                        count = count +1;
                    end
                    strout = [ strout strout2 ];
                    power.count = power.count + 1;

                case 133
                    % 3 channel data
                    if opt.nbchan == 0
                        opt.nbchan = 3;
                        opt.command = 'reset';
                        set(fig, 'userdata', opt);
                    else
                        data.values(1,data.count) = payloadBuffer(3)*256+payloadBuffer(4);
                        data.values(2,data.count) = payloadBuffer(5)*256+payloadBuffer(6);
                        data.values(3,data.count) = payloadBuffer(7)*256+payloadBuffer(8);
                        if ~isempty(fid1), 
                            strout = sprintf([strout ' %.2X %.2X %.2X %.2X %.2X %.2X'], payloadBuffer(3), payloadBuffer(4), ...
                                payloadBuffer(5), payloadBuffer(6), payloadBuffer(7), payloadBuffer(8));
                        end;
                        data.time(data.count) = toc;
                        data.count = data.count + 1;
                    end;

                case 144  % hex 90 - accel data
                    if opt.nbchan_a == 0
                        opt.nbchan_a = 3;
                        opt.command = 'reset';
                        set(fig, 'userdata', opt);
                    else
                        % 3 channel data
                        data2.values(1,data2.count) = payloadBuffer(3)*256+payloadBuffer(4);
                        data2.values(2,data2.count) = payloadBuffer(5)*256+payloadBuffer(6);
                        data2.values(3,data2.count) = payloadBuffer(7)*256+payloadBuffer(8);
                        if ~isempty(fid1), 
                            strout = sprintf([strout ' %.2X %.2X %.2X %.2X %.2X %.2X'], payloadBuffer(3), payloadBuffer(4), ...
                                payloadBuffer(5), payloadBuffer(6), payloadBuffer(7), payloadBuffer(8));
                        end;
                        data2.count = data2.count + 1;
                    end;

                otherwise
                    if ~isempty(fid1), 
                        for index = 2:length(payloadBuffer)
                            strout = sprintf([strout ' %.2X'], payloadBuffer(index));
                        end;
                    end;
            end;

            % print out the Thinkgear line
            if ~isempty(fid1)
            %    try fprintf(fid1, '%s\n', strout); catch end;
            %end;
                if payloadBuffer(1) == 128
                    try 
                        % 從 data 結構中取得剛剛解析好的最新數值
                        % data.values 是 double 格式，我們存成整數即可
                        current_raw_value = data.values(data.count-1);
                        fprintf(fid1, '%d\n', current_raw_value);
                        %fprintf(fid1, '%s\n', strout); 
                        
                        % --- 新增：計數與換檔邏輯 ---
                        opt.line_count = opt.line_count + 1; % 增加計數
                        
                        if opt.line_count >= 64
                            % 1. 關閉目前的檔案
                            fclose(fid1);
                            if ~isempty(fid2), fclose(fid2); end;

                            % --- 新增：寫入指標檔 (Handshake) ---
                            % 告訴 Python: "opt.file_seq" 這個檔案已經 Ready 了
                            try
                                fid_flag = fopen('data_ready.txt', 'w');
                                fprintf(fid_flag, '%d', opt.file_seq); 
                                fclose(fid_flag);
                            catch end
                            % ---------------------------------
                            
                            % 2. 重置計數，增加檔案序號
                            opt.line_count = 0;
                            opt.file_seq = opt.file_seq + 1;
                            if opt.file_seq > 8
                                opt.file_seq = 1; % 超過 8 就回到 1，進行覆寫
                            end
                            
                            % 3. 產生新的檔名 (加入 _partX 序號)
                            % 這裡我們複製並修改了原本 case 'save' 的檔名生成邏輯
                            if opt.save_thinkgear
                                %if opt.dateinfilename
                                filename_t = sprintf('%s_part%d_thinkgear.txt', opt.basefilename, opt.file_seq);
                                %else
                                %    filename_t = sprintf('%s_part%d_thinkgear.txt', opt.basefilename, opt.file_seq);
                                %end;
                                fid1 = fopen(filename_t, 'w'); % 開啟新檔
                                opt.fid_thinkgear = fid1;      % 更新全域結構中的 handle
                                opt.last_saved_file = filename_t; % 更新最後存檔名以便 stop 時存 event
                            end
                            
                            % 4. 如果有開 Stream log，也要跟著換新檔
                            if opt.save_stream
                                %if opt.dateinfilename
                                %    filename_s = sprintf('%s_part%d_stream_%s%s%s.txt', opt.basefilename, opt.file_seq, ...
                                %                    datestr(now, 'ddmmmyyyy_HHhMM'), 'm', datestr(now,'SS'));
                                %else
                                % --- remove time stamp ---
                                filename_s = sprintf('%s_part%d_stream.txt', opt.basefilename, opt.file_seq);
                                %end;
                                fid2 = fopen(filename_s, 'w');
                                opt.fid_streamlog = fid2;
                            end
                            
                            % 更新 opt 到 figure userdata (雖非必要但建議)
                            set(fig, 'userdata', opt);
                            fprintf('File split: Starting part %d\n', opt.file_seq); % 在 Command Window 顯示提示
                        end
                    % --- 結束新增邏輯 ---
                    catch end
                end
            end
        end
    end

    % plot a specific position
    % ------------------------
    if strcmpi(opt.command, 'plotposition')
        if  lastdatapos ~= opt.datapos;
            endpos   = round((data.count -opt.winlen  )*opt.datapos)+opt.winlen  +1;
            endpos_a = round((data2.count-opt.winlen_a)*opt.datapos)+opt.winlen_a+1;
            plotnow = 1;
        else
            plotnow = 0;
        end;
        lastdatapos = opt.datapos;
    else
        endpos     = data.count  -1;
        endpos_a   = data2.count-1;
        plotnow = 0;
    end;

    % attention.count starts at 1, increments per second
    % endpos
    endpos_pan = min(floor(endpos/opt.srate), attention.count-1);

    %%%%%%%%%%%%%%%%%
    %%% PLOT DATA %%%
    %%%%%%%%%%%%%%%%%

    % roughly updates every 0.2 seconds (measured about 0.25 seconds)
    if toc-plotDelay > opt.refreshrate || plotnow == 1

        if data.count-1 > opt.winlen 
            nbchan = opt.nbchan+opt.nbchan_a;

            %plotdata(opt.hdl_axisplot, nbchan, 0, data.time, data.values, data.srate, endpos, opt.winlen, opt.scale, opt.scalevert, opt.event, opt.firmwarecorrect, bitlevels);

            plotdata(opt.hdl_axisplot, nbchan, 0, data.time, data.values, data.srate, endpos, opt.winlen, opt.scale, opt.scalevert, opt.event, opt.firmwarecorrect, bitlevels);
            if opt.nbchan_a > 0
                plotdata(opt.hdl_axisplot, nbchan, data.nbchan, data.time, data2.values, data2.srate, endpos_a, opt.winlen_a, opt.scale, opt.scalevert, opt.event, opt.firmwarecorrect, bitlevels);
            end;
            %drawnow;

            plotDelay = toc;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % play the actual wave plot as sound %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             finalsound = [];
%             
%             for i=1:30
%                 f=1*(data.values(endpos-30+i)+1);   % take one value (0-255)
%                 rad = 0:1/60:2;
%                 a = sin(2*pi*f*rad);
%                 
%                 finalsound = [finalsound a];
%             end
%                     
%             soundsc(finalsound, 20000);
        end
    end

    if (toc-plotDelaySlider > sliderrefreshrate)
        plotDelaySlider = toc;        
        if strcmpi(opt.command, 'continue')
            steps = min(opt.srate/data.count,1);
            if get(slider, 'userdata')
                set(slider, 'sliderstep', [steps/10 steps], 'value', 1);
            end;
        end;
        if data.count > data.srate*10
            sliderrefreshrate = 3;
        end;
    end;
    
    %%%%%%%%%%%%%%%%
    %%% PLOT FFT %%%
    %%%%%%%%%%%%%%%%
    % if the time has elapsed then replot the FFT
    if (toc-plotDelayFFT > opt.fftrefreshrate) && endpos > min(2*opt.fftwinlen, 2*opt.srate) 
        plotDelayFFT = toc;

        if 0
            b = now;
            fprintf('Debug msg: time elpased (s) %2.2f | port buffer wait:%s | packets %d\n', ...
                (b-a)*100000, fastif(waitflag, 'yes', 'no'), length(allpayloadBuffer) );
            a = b;
            %toc, tic;
        end;

        lastseconddata = data.values(endpos-min(opt.fftwinlen, opt.srate)+1:endpos);

        rawfft128 = fft(lastseconddata/1023*6-3, opt.srate);
        freqs     = linspace(0, opt.srate/2, size(rawfft128,2)/2);
        rawfft128 = rawfft128(2:length(rawfft128)/2+1);
        powertmp  = rawfft128.*conj(rawfft128);
        indnon0   = find(powertmp ~= 0);
        rawfft128(indnon0) = powertmp(indnon0)/opt.srate;
        rawfft128(find(powertmp == 0)) = NaN;

        % remove frequencies above 64
        freqs     = freqs(1:64);
        rawfft128 = rawfft128(1:64);

        rawfftplot = (rawfft128(1:2:end)  + rawfft128(2:2:end))/2;
        freqsplot  = (freqs(1:2:end)      + freqs(2:2:end)    )/2;
        rawfftplot = (rawfftplot(1:2:end) + rawfftplot(2:2:end))/2;
        freqsplot  = (freqsplot(1:2:end)  + freqsplot(2:2:end)    )/2;
        if opt.fftlog
            rawfftplot = 10*log10(rawfftplot);
        end

        %rawfftplot = rawfft128;
        %freqsplot = freqs;

        if ~strcmpi( opt.fftplottype, '3d')
            plotspectrum( opt.hdl_axisfft, opt.fftplottype, freqsplot, rawfftplot, opt.fftscale);
        end;

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% STORE FFT AT REGULAR INTERVALS %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (toc-storeDelayFFT > opt.fftwinlen/opt.srate && min(opt.fftwinlen, opt.srate) && exist('rawfft128', 'var')) || plotnow == 1    

        storeDelayFFT = toc;
        if ~strcmpi(opt.command, 'plotposition')
            ffthist.values(ffthist.count,:) = rawfft128(1:64);
            endpos_fft    = ffthist.count;
            ffthist.count = ffthist.count+1;
        else
            endpos_fft = floor(endpos/opt.fftwinlen);
        end;
        if strcmpi( opt.fftplottype, '3d')
            ffthistrange = endpos_fft-19:endpos_fft;
            if ffthistrange(1)<1, ffthistrange = ffthistrange-ffthistrange(1)+1; end;
            if opt.fftlog
                 rawfftplot = 10*log10(ffthist.values(ffthistrange,:)');
            else rawfftplot = ffthist.values(ffthistrange,:)';
            end
            plotspectrum( opt.hdl_axisfft, '3d', freqs, rawfftplot, opt.fftscale);
        end;

    end;

    %%%%%%%%%%%%%%%%%%
    %%% PLOT PANEL %%%
    %%%%%%%%%%%%%%%%%%
    for pan = 1:length(opt.panaxis)     % goes from 1 to 2
        if toc-plotDelaypanel(pan) > opt.panrefreshrate(pan) && exist('endpos_fft', 'var') || plotnow == 1
            plotDelaypanel(pan) = toc;

            if strcmpi(opt.pancontent{pan}, 'None')

                plotpanel(opt.panaxis(pan), pan, [], { 'Use menu to' 'select measure' 'to plot' });

            elseif strcmpi(opt.pancontent{pan}, 'Custom freq.')

                datapan = computepower( 3.3*(data.values(endpos+1-opt.srate:endpos)-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/opt.firmwarecorrect, opt.srate, ...
                    opt.panfreqlimits{pan}, opt.panwinlen(pan), opt.panrefreshrate(pan));
                titletxt = sprintf('Custom freq. (dB) %2.1f-%2.1f', opt.panfreqlimits{pan}(1), opt.panfreqlimits{pan}(2));
                times = linspace(endpos+1-opt.panwinlen(pan), endpos, length(datapan));
                plotpanel(opt.panaxis(pan), pan, times/opt.srate, datapan, opt.panscale{pan}, titletxt);                

            elseif strcmpi(opt.pancontent{pan}, 'Custom func.')

                % opt.panscale{pan} 
                titletxt = [ opt.pancontent{pan} ' "' opt.panfunction{pan} '"' ];
                if(endpos+1-opt.panwinlen(pan) > 1)
                    try
                        data.values(endpos+1-opt.panwinlen(pan):endpos);
                        datapan = feval(str2func(opt.panfunction{pan}), ...
                             3.3*(data.values(endpos+1-opt.panwinlen(pan):endpos)-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/opt.firmwarecorrect, ...
                             opt.srate);
                        times = linspace(endpos+1-opt.panwinlen(pan), endpos, length(datapan));
                        plotpanel(opt.panaxis(pan), pan, times/opt.srate, datapan, opt.panscale{pan}, titletxt);                
                    catch,
                        plotpanel(opt.panaxis(pan), pan, [], { 'Error running' 'the custom' 'function' });
                        disp([ 'Error running custom function: ' lasterr ]);
                    end;
                else
                    plotpanel(opt.panaxis(pan), pan, [], { 'Not enough' 'data to plot' 'yet' });
                end
            else

                datapan = [];
                %datapan2 = [];
                %dtaapan3 = [];

                switch opt.pancontent{pan}
                  case 'Attention'   , datapan = attention;  chan_ind = 1; eegpowers = 0; titletxt = 'Attention';
                  case 'Meditation'  , datapan = meditation; chan_ind = 1; eegpowers = 0; titletxt = 'Meditation';
                  case 'EEG delta'   , power.values;      chan_ind = 1; eegpowers = 1; titletxt = 'delta';
                  case 'EEG theta'   , power.values;      chan_ind = 2; eegpowers = 1; titletxt = 'theta';
                  case 'EEG alpha1'  , power.values;      chan_ind = 3; eegpowers = 1; titletxt = 'alpha1';
                  case 'EEG alpha2'  , power.values;      chan_ind = 4; eegpowers = 1; titletxt = 'alpha2';
                  case 'EEG beta1'   , power.values;      chan_ind = 5; eegpowers = 1; titletxt = 'beta1';
                  case 'EEG beta2'   , power.values;      chan_ind = 6; eegpowers = 1; titletxt = 'beta2';
                    otherwise, eegpowers = 0;
                end;
                % opt.panwinlen(1/2) = 1024?   opt.srate = 128/256    datapan.srate = 1
                %                                        1024/128*1 = 8
                % round(opt.panwinlen(pan)/opt.srate*datapan.srate)

                if (eegpowers == 1)
                    %fprintf('Plotting EEG powers\n');
                    nvals = round(opt.panwinlen(pan)/opt.srate);
                    endpos_beg  = max(power.count-nvals,1);

                    %a = power.values(1:8,1:30)     % show all eeg data powers
                    %size(power.values)
                    %size(datapan2.values)      

                    %b = datapan.values(1:8,1:30);
                    %b
                    % pull the data out into datapan
                    datapan3 = power.values(chan_ind, endpos_beg:(power.count-1));

                    % pad it with zeros if we don't have enough data
                    if length(datapan3) < 3
                        plotpanel(opt.panaxis(pan), pan, [], { 'Measure' 'values are not' 'available' });
                    else
                        datapan3 = [ zeros(1,nvals-length(datapan3)) datapan3 ];
                        times = linspace(endpos_beg, power.count, nvals);
                        plotpanel(opt.panaxis(pan), pan, times, datapan3, opt.panscale{pan}, titletxt);
                    end;

                else
                    % fprintf('Plotting Att/Med values\n');

                    endpos_beg  = max(endpos_pan-round(opt.panwinlen(pan)/opt.srate*datapan.srate),1);
                    datapan     = datapan.values(chan_ind, endpos_beg:endpos_pan);
                    if length(datapan) < round(opt.panwinlen(pan)/opt.srate)
                        datapan = [ zeros(1,opt.panwinlen(pan)/opt.srate-length(datapan)) datapan ];
                    end;
                    times = linspace(endpos_pan*opt.srate-opt.panwinlen(pan), endpos_pan*opt.srate, length(datapan));
%                     opt.panaxis(pan)
%                     pan
%                     times/opt.srate
%                     datapan
%                     opt.panscale(pan)
%                     titletxt
                    plotpanel(opt.panaxis(pan), pan, times/opt.srate, datapan, opt.panscale{pan}, titletxt);

                end;

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % play the attention number as a sound %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 %if strcmp(opt.pancontent{pan}, 'Attention')
%                 if strcmp(opt.pancontent{pan}, 'Meditation')
%                     datapan(end)
%                     f = 50*(datapan(end)+1);
%                     rad = 0:1/4000:1;
% 
%                     a = sin(2*pi*f*rad);
%                 end
%                 
%                 soundsc(a, 10000);

            end;
        end;
    end;
end

% close files and figure
% ----------------------
if ~isempty(opt.fid_thinkgear)
    neuroskylab('stop', fig);
end;
delete(fig);
try,
    save('-mat', '-V6', 'options.mat', 'opt');
catch, end;

if strcmpi(opt.command, 'transfer')
    if isfield(opt,'last_saved_file')
        if ~isempty(opt.last_saved_file)
            disp('Import last saved file');
            try
                evalin('base', 'eeglab redraw');
            catch
                error('Cannot find EEGLAB, check your path definition');
            end;
            if ~exist('pop_loadthinkgear')
                disp('To export data to EEGLAB, you must first install this software');
                disp('as well as the thinkgear plugin (see documentation for details)');
            else
                try, 
                    evalin('base', [ 'EEG = pop_loadthinkgear(''' opt.last_saved_file ''');' ] );
                    evalin('base', '[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);' );
                    evalin('base', 'eeglab redraw;' );            
                catch
                    error('Error while importing file');
                end;
            end;
        else
            disp('Neuroskylab will only export saved data to EEGLAB. No file saved.');
        end;
    else
        disp('Neuroskylab will only export saved data to EEGLAB. No file saved.');
    end;
end;

% -------------------------------------------------------------------
% function to plot scale (indicator mark next to eeg data) (not used)
% -------------------------------------------------------------------
% calling: plotscale(opt.scale/opt.firmwarecorrect, opt.scalevert/opt.firmwarecorrect, opt.nbchan)

function plotscale(scale, scalevert, nbchan)

persistent hdl_scale;
if ~isempty(hdl_scale), delete(hdl_scale); end;

scale = round(scale*1000000/10)*10;
if nbchan ==  1
    hdl_scale(1) = plot([0.5 0.5], [0 scale], 'k'); hold on; % draw a vertical line
    hdl_scale(2) = plot([0.4 0.6], [0 0], 'k');             % draw lower horz line
    hdl_scale(3) = plot([0.4 0.6], [scale scale], 'k');     % draw upper horz line
    ylim([0 scale*4]); % for visual purpose, scale -> 1/4 height
    hdl_scale(4) = text(0.5, scale*1.1, [ num2str(scale,3) ' \muV'], 'interpreter', 'tex', 'rotation', 90, 'fontsize', 8);
else
    scale = round(scalevert*10000)/10000;
    hdl_scale(1) = plot([0.5 0.5], [0 scale], 'k'); hold on;
    hdl_scale(2) = plot([0.4 0.6], [0 0], 'k');
    hdl_scale(3) = plot([0.4 0.6], [scale scale], 'k');
    ylim([-0.5*scalevert (nbchan+0.5)*scalevert]);
    hdl_scale(4) = text(0.5, scale*1.1, [ num2str(scale*1000000,3) ' \muV'], 'interpreter', 'tex', 'rotation', 90, 'fontsize', 8);
end;
xlim([0.2 0.8]);
axis off;

% -------------------------------------------------------
% function to plot panel (user selectable processed data)
% -------------------------------------------------------
function plotpanel(hdl_axis, pan, times, data, panscale, titletxt )

%if ~isequal( get( hdl_axis, 'parent'), gcf), return; end;

persistent hdl_plot;
persistent txt_flag;
persistent oldpanscale;
if isempty(txt_flag), txt_flag = [0 0]; end;
if isempty(hdl_plot)   , hdl_plot    = { [] [] }; end;
if isempty(oldpanscale), oldpanscale = { [] [] }; end;

if isempty(times)
    if txt_flag(pan) == 0
        hdl_plot{pan} = []; 
        oldpanscale{pan} = { [] }; 
        axes(hdl_axis); cla;
        set(gca, 'xticklabel', [], 'yticklabel', []); box on;
        ylabel('V^2');
        text(0.15, 0.7, data{1}, 'unit', 'normalized', 'fontsize', 15, 'fontweight', 'bold', 'color', [.8 .8 .8]);
        text(0.15, 0.5, data{2}, 'unit', 'normalized', 'fontsize', 15, 'fontweight', 'bold', 'color', [.8 .8 .8]);
        text(0.15, 0.3, data{3}, 'unit', 'normalized', 'fontsize', 15, 'fontweight', 'bold', 'color', [.8 .8 .8]);
        title('');
        txt_flag(pan) = 1;
    end;
    return;
end;
txt_flag(pan) = 0;

if isempty(hdl_plot{pan}) || ~isequal(oldpanscale{pan}, panscale)
    axes(hdl_axis);
    hdl_plot{pan} = plot(times, data);
    if ~isempty(panscale), ylim(panscale);
    else yl = ylim; ylim(yl); 
    end;
    xlim([times(1) times(end)]);
    title(titletxt);
else
    set(hdl_plot{pan}, 'xdata', times, 'ydata', data);
    set(hdl_axis, 'xlim', [times(1) times(end)]);
    if isempty(panscale)
        
        yl = ylim;
        
            % Auto scaling
%         if any(data > yl(2)) || any(data < yl(1))
%             yl(1) = min(min(data), yl(1));
%             yl(2) = max(max(data), yl(2));
%         elseif mean(abs(data)) < yl(2)/3
%             yl(1) = min(data);
%             yl(2) = max(data);
%         end;
        set(hdl_axis, 'ylim', yl);
    end;
end;
oldpanscale{pan} = panscale;

% -------------------------
% function to plot spectrum
% -------------------------
function plotspectrum( hdl_axis, fftplottype, freqs, fftdata, fftscale )

%if ~isequal( get( hdl_axis, 'parent'), gcf), return; end;

persistent hdl_plot;
persistent oldfftscale;
if isempty(oldfftscale), oldfftscale = [-Inf Inf]; end;
if isempty(fftplottype), delete(hdl_plot); hdl_plot = []; end;

if strcmpi(fftplottype, '3D')

    fftdata = (fftdata(1:2:end,:) + fftdata(2:2:end,:))/2; % divide resolution by 2
    freqs   = (freqs(1:2:end)     + freqs(2:2:end)    )/2; % otherwise display glitches
    if ~isequal( get( hdl_axis, 'parent'), gcf), return; end;
    
    tmpgca = gca;
    axes(hdl_axis);
    if ~strcmpi(get(hdl_plot, 'type'), 'patch'), delete(hdl_plot); hdl_plot = []; end;
    
    if ~isempty(hdl_plot), delete(hdl_plot); end;
    timevals = 0:19;
    hdl_plot = mysurf2(timevals, freqs, fftdata);
    set(hdl_axis, 'xdir', 'reverse');
    zlim( fftscale);
    caxis(fftscale);
    xlim([timevals(1) timevals(end)]);
    ylim([freqs(1) freqs(end)]);
    view(147, 73);
    set(hdl_axis, 'ytick', [0 20 40 60], 'xtick', [0 5 10 15]);
    set(hdl_axis, 'yticklabelmode', 'auto', 'xticklabelmode', 'auto');
    t = ylabel('Frequencies'); set(t, 'rotation', -45);
    t = xlabel('Seconds');     set(t, 'rotation', 25);
    axes(tmpgca);
    %surf(1:20, freqs(1:maxfreq), ffthist.values(:,1:maxfreq)');
    %set(gca, 'xdir', 'reverse');
    %set(gca, 'ydir', 'reverse');
    %ylim([freqs(1) freqs(maxfreq)]);
    %[az el] = view;
    %view([az el+30]);
    
elseif strcmpi(fftplottype, 'hist')

    if ~strcmpi(get(hdl_plot, 'type'), 'hggroup'), delete(hdl_plot); hdl_plot = []; end;    
    
    % lower plot resolution
    if mod(length(freqs),2) == 1,
        freqs(end)   = [];
        fftdata(end) = [];
    end;
    %freqs   = (freqs(1:2:end-1)+freqs(2:2:end))/2;
    %fftdata = (fftdata(1:2:end-1)+fftdata(2:2:end))/2;

    if isempty(hdl_plot) | any(fftscale ~= oldfftscale)
        axes(hdl_axis);
        hdl_plot = bar(freqs,60+fftdata);
        set(hdl_axis, 'ytickmode', 'auto');
        axis([freqs(1) freqs(end) fftscale+60 ]);
        ytickl = get(hdl_axis, 'yticklabel');
        ytick  = get(hdl_axis, 'ytick');
        set(hdl_axis, 'ytickmode', 'manual');
        set(hdl_axis, 'yticklabelmode', 'manual' );
        ytickl = linspace(fftscale(1), fftscale(2), length(ytick));
        set(hdl_axis, 'ytick', ytick, 'yticklabel', ytickl);
        title('Frequency Spectrum');
        ylabel('Power in \muV^{2}');
        xlabel('Hertz');
    else
        set(hdl_plot, 'xdata', freqs, 'ydata', 60+fftdata);
    end;
    
else

    if ~strcmpi(get(hdl_plot, 'type'), 'line'), delete(hdl_plot); hdl_plot = []; end;    

    if isempty(hdl_plot) | any(fftscale ~= oldfftscale)
        axes(hdl_axis);
        hdl_plot = plot(freqs,fftdata);
        title('Frequency Spectrum');
        ylabel('Power in dB (0dB = 50\muV)');
        xlabel('Hertz');
        ylim( fftscale );
        xlim([freqs(1) freqs(end) ]);
    else
        set(hdl_plot, 'xdata', freqs, 'ydata', fftdata);
    end;
end;
oldfftscale = fftscale;

% ----------------------------------------
% function to plot data (main eeg raw data)
% ----------------------------------------

% calling function for single channel
% plotdata(opt.hdl_axisplot, nbchan, 0, data.time, data.values, data.srate, 
%          endpos, opt.winlen, opt.scale, opt.scalevert, opt.event, opt.firmwarecorrect);

function plotdata(hdl_axis, nbchan, offset, timeori, data, srate, endpos, winlen, scale, scalevert, event, gain, bitlevels)

persistent hdl_plot;
persistent hdl_event;
%fprintf('X');

%if ~isequal( get( hdl_axis, 'parent'), gcf), return; end;

% in time stamp (x axis) is missing, delete plot object and make a new one
if isempty(timeori), delete(hdl_plot); hdl_plot = []; return; end;
scalevert = scalevert/gain;
     
if mod(endpos,2) == 1, endpos = endpos - 1; end; 
time_inds    = endpos-winlen+1:endpos;
colors       = { 'r' 'b' 'g' 'm' 'c' 'k' };
if nbchan == 1
    % scale    = scale * 2; % see plotscale -> for visual purpose, scale=1/4 height
    
    if ~isempty(hdl_plot)
        if srate == 512
            set(hdl_plot(1), 'xdata', time_inds(1:2:end)/srate, 'ydata', 3.3*(data(time_inds(1:2:end))-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/gain);
        else
            set(hdl_plot(1), 'xdata', time_inds/srate, 'ydata', 3.3*(data(time_inds)-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/gain);
        end;
    else
        axes(hdl_axis); cla; hold on;
        if srate == 512
            hdl_plot = plot(time_inds(1:2:end)/srate, 3.3*(data(time_inds(1:2:end))-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/gain,'r'); hold on;
        else
            hdl_plot = plot(time_inds/srate, 3.3*(data(time_inds)-mean(bitlevels))/(bitlevels(2)-bitlevels(1))/gain,'r'); hold on;
        end;
        ylabel('Volts');
        set(hdl_axis, 'yticklabelmode', 'auto', 'ylimmode', 'manual');
    end;
    
    set(hdl_axis, 'xlim', [(endpos-winlen)/srate endpos/srate ], 'ylim', [-scale scale]/gain*2);

%    tmpl = str2double(get(hdl_axis, 'yticklabel'));
%    set(hdl_axis, 'yticklabelmode', 'manual', 'yticklabel', round(tmpl*100));
    %xlim([(endpos-winlen)/srate endpos/srate ]);
    %ylim([-scale scale]);
    
else
    % note nbchan gets set to other than 1 somehow even with 1 channel mode
    minord    = -scalevert;
    maxord    = scalevert;% + (nbchan-1)*2;
    
    
    if length(hdl_plot) > offset && hdl_plot(offset+1) ~= 0
        for i = 1:size(data,1)
            set(hdl_plot(i+offset),'xdata', time_inds/srate,'ydata', ((data(i, time_inds)*3/1023-1.5)/gain/scale + scalevert*(i+offset-0.5)/scale));
        end;
    else
        axes(hdl_axis); 
        if offset == 0, cla; hold on; disp('test'); end;
        for i = 1:size(data,1)
            hdl_plot(i+offset) = plot(time_inds/srate, ((data(i, time_inds)*3/1023-1.5)/gain/scale + scalevert*(i+offset-0.5)/scale), colors{i+offset}); hold on;
        end;
    end;
    set(hdl_axis, 'xlim', [(endpos-winlen)/srate endpos/srate], 'ylim', [minord maxord]);
    set(hdl_axis, 'yticklabelmode', 'manual');
    % [-3 3]/gain);
    % [minord maxord]);
    % pause;
end;
set(hdl_axis, 'xticklabelmode', 'auto');

% plot event lines
delete(hdl_event);
hdl_event = [];
if ~isempty(event)
    evt.count = 0;
    cont      = 1;
    while cont && evt.count < length(event)
        if event(end-evt.count).latency < timeori(time_inds(end)) && event(end-evt.count).latency > timeori(time_inds(1)) 
            if ~isequal(gca, hdl_axis), axes(hdl_axis); end;
            [tmp index] = min(abs(timeori(time_inds)-event(end-evt.count).latency));
            hdl_event(end+1) = plot([time_inds(index) time_inds(index)]/srate, [-100 100], 'b');
            yl = ylim; xl = xlim;
            hdl_event(end+1) = text(time_inds(index)/srate-(xl(end)-xl(1))*0.015, yl(end)+(yl(end)-yl(1))*0.05, event(end-evt.count).type, 'color', 'b');
        elseif event(end-evt.count).latency <= timeori(time_inds(1))
            cont = 0;
        end;
        evt.count = evt.count + 1;
    end;
end;
%xlabel('Time (s)');
%set(gca, 'yticklabel', []);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Read one Packet from Port or File %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [payloadBuffer, A, waitflag ] = read_packets( fileid_replay, filterbitstream )

    persistent remain;
    persistent tmptime;
    if isempty(tmptime), tmptime = toc; end;
    %gfprintf('.');
    
    global s;
    payloadBuffer = {};
    
    inds     = [];
    waitflag = 0;
    while length(inds) < 2
        
        if ~isempty( fileid_replay )
            try
                % read one line from streamlog file and decode it
                % -----------------------------------------------
                tmpline   = fgetl( fileid_replay );
                if isempty(tmpline), tmpline   = fgetl( fileid_replay ); end;
                tmpind    = find(tmpline == ':');
                tmpbuffer = tmpline(tmpind+1:end);
                inds      = find(tmpbuffer == ' ');
                if length(inds) > 1
                    if inds(1) == 1 && inds(2) == 2, inds(1) = []; end;
                end;
                A         = zeros(length(inds), 1);
                for i = 1:length(inds)
                    A(i) = hex2dec(tmpbuffer(inds(i)+1:inds(i)+2));
                end;
            catch,
                fseek(fileid_replay,0,1);
                A = [];
                warndlg('Wrong file format');
                return;
            end;
        else % fileid_replay empty so read from port
            % read bytes from port
            % -------------------
            
            % wait up to 1 second until bytes are read to read
            count = 0; while s.BytesAvailable == 0 && count < 300, pause(0.05); count = count+1; waitflag = 1; end;
            
            % 1 second past, generate an error
            if count == 300, delete(gcf); error('Port time out'); end;
            
            % read everything into A
            A = fread(s, s.BytesAvailable);
            if ~isempty(filterbitstream)
                A = filterbitstream(A);
            end;
            
            %for i=1:length(A), fprintf('%.2X ', A(i)); end; fprintf('\n', A(i));
            if length(A) == 2048, disp('Max buffer length, some packet might have been lost'); end;
        end;
        
        % concatenate with previous bytes
        A = [ remain; A ];
        
        % find all indicies of AA AA beginning on odd positions
        inds = strfind(A', [170 170]);
        
        % remove any extra AA before the AA AA
        inds(find(inds(1:end-1) == inds(2:end)-1)) = [];
        
        if length(inds) < 2,    % we did not get a full packet yet
            remain = A;         % all of it is a remainder so put it all in
        end;
    end;

    % segment the outputs
    % -------------------
    count = 1;
    for i = 1:length(inds)-1
        % Make sure beginning of AA packets are separated by at least 2 bytes
        if inds(i) ~= inds(i+1)-2 
            % look at a single packet that begins with AA AA
            [ tmpbuf checksum ] = checkpacket(A(inds(i):inds(i+1)-1));
            if ~checksum
                %disp('Warning: packet integrity lost due to discontinuity in acquisition');
            else
                % tmpbuf has a byte number (0-255) in each row
                % tmpbuf
                parsedpackets = parse_packet( tmpbuf );
                
                % parsedpackets has the whole array
                
                % copy valid packets to the payloadBuffer
                for p = 1:length(parsedpackets) % this will only copy the first cell over
                    payloadBuffer{count} = parsedpackets{p};
                    count = count+1;
                end;
            end;
        end;
    end;
    
    % put the left over packets in the remain variable
    remain = A(inds(end):end);
    
    % clear out partial packets in the raw packets 
    A(inds(end):end) = [];

    % if datafile -> wait for smooth scrolling
    if ~isempty( fileid_replay )
        expectedtime = length(A)/512/8;
        pause(expectedtime-(toc-tmptime));
        tmptime = toc;
    end;
        
function [parsedpacket] = parse_packet( packet )

    packet_length = length( packet );
    packet_index = 1;
    datarow_index = 1;
    parsedpacket = {};
    
    % While there are more DataRows in the packet to process...
    while( packet_index < packet_length ) 

        % Determine the EXCODE level of this DataRow's CODE by counting 
        % the number of EXCODE bytes preceding the CODE
        % (EXCODE bytes are bytes of value 0x55; CODE is never 0x55)
        datarow_excode_level = 0;
        while( packet(packet_index) == 85 )
            datarow_excode_level = datarow_excode_level + 1;
            packet_index = packet_index + 1;
        end

        % Determine the CODE of this DataRow
        datarow_code = packet( packet_index );
        packet_index = packet_index + 1;

        % Determine the number of bytes comprising this DataRow's value
        % (CODEs less than 0x80 are single-byte values, while CODEs greater
        % than or equal to 0x80 are multi-byte values.  For multi-byte values, the 
        % length of the multi-byte value immediately follows the CODE)
        if( datarow_code >= 128 )
            value_length = packet( packet_index );
            if value_length+packet_index <= length(packet) % account for some currupted packets
                parsedpacket(datarow_index) = { packet(packet_index-1:(packet_index + value_length)) };
            end;
            packet_index = packet_index + value_length + 1;
        else
            parsedpacket(datarow_index) = { packet(packet_index-1:packet_index) };
            packet_index = packet_index + 1;
        end

        % Done with the current DataRow
        datarow_index = datarow_index + 1;
        
end % "Process each DataRow..."
    
% check a single packet
% ---------------------
function [payloadBuffer, checksumres] = checkpacket(  payloadBuffer )
    
    % read data into buffer
    % ---------------------
    rawchecksum = payloadBuffer(end);
    payloadBuffer = payloadBuffer(4:end-1);
    payloadSum = sum(payloadBuffer(1:length(payloadBuffer))); %payloadSum

    % read checksum
    % -------------
    checksum = bitand(rawchecksum, 255);
    %checksum = dec2bin(rawchecksum);
    %if length(checksum) > 8                 %if checksum is more than 8 bits, get the lower 8 bits
    %    checksum = checksum(length(checksum)-7:length(checksum));
    %end
    %checksum = bin2dec(checksum);
    checksum = 2^8 - 1 - checksum;          %compute checksum 1's inverse

    % compute data checksum
    % ---------------------
    payloadSum = bitand(payloadSum, 255);
    %payloadSum = dec2bin(payloadSum);
    %if length(payloadSum) > 8               %if payloadsum is more than 8 bits, get the lower 8 bits
    %    payloadSum = payloadSum(length(payloadSum)-7:length(payloadSum));
    %end
    %payloadSum = bin2dec(payloadSum);

    checksumres = checksum == payloadSum;

% ---------------------
% decode gui parameters
% ---------------------
function res = getguioutput(fig)
allobj = findobj('parent', fig);
counter = 1;
res = [];
for index=1:length(allobj)
    tmptag = get(allobj(index), 'tag');
    if ~isempty(tmptag)
        if strcmpi(get(allobj( index ), 'style'), 'edit')
             res = setfield(res, { counter }, tmptag, get( allobj( index ), 'string'));
        else res = setfield(res, { counter }, tmptag, get( allobj( index ), 'value'));
        end;
    end;
end;    

% ---------------------
% get eeg power
% ---------------------
function [times, outvar] = geteegpower( freqs, ffthist, fftpos, srate, panfreqlimits, fftwinlen, panwinlen)

nfftwin = floor(panwinlen/fftwinlen); if nfftwin == 0, nfftwin = 1; end;

% copy frequencies
[tmp minf] = min( abs(freqs - panfreqlimits(1)) );
[tmp maxf] = max( abs(freqs - panfreqlimits(2)) );
bandpower = mean(ffthist(max(1, fftpos-nfftwin+1):fftpos, minf:maxf),2);
if length(bandpower) < nfftwin, bandpower = [ zeros(1, nfftwin-length(bandpower))*NaN bandpower']; end;

% generate time axis
times = linspace((fftpos-nfftwin)*fftwinlen, fftpos*fftwinlen, nfftwin);
outvar = bandpower;

% --------------------------------------------------------------
% 4-byte hex to float conversion
% (standard Matlab only allows 8-byte double)
% adapted from Excel file "ieeefloats.xls" found on the internet
% --------------------------------------------------------------
% called using: power.values(power.count,count) = hex2float(hexstr)
function f = hex2float(hexstr)

    num = uint32(hex2dec(hexstr));

    sign = fastif(bitget(num,32), -1, 1);

    expo  = num;
    for i=[1:23 32], expo = bitset(expo, i, 0); end;
    expo  = double(bitshift(expo, -23))-127;

    val  = num;
    for i=24:32, val = bitset(val, i, 0); end;
    val = 1+double(val)/8388608;

    f = double(sign)*val*2^double(expo);    
    

%%
% Parses @c payload into an array of @c datarows (CODE + value) according to the ThinkGear
% Packet specification.
%
function [datarows] = parse_payload( payload )

    payload_length = length( payload );
    payload_index = 1;

    datarows = {};
    datarow_index = 1;
   
    % While there are more DataRows in the packet to process...
    while( payload_index < payload_length )

        % Determine the EXCODE level of this DataRow's CODE by counting
        % the number of EXCODE bytes preceding the CODE
        % (EXCODE bytes are bytes of value 0x55; CODE is never 0x55)
        datarow_excode_level = 0;
        while( payload(payload_index) == 85 )
            datarow_excode_level = datarow_excode_level + 1;
            payload_index = payload_index + 1;
        end

        % Determine the CODE of this DataRow
        datarow_code = payload( payload_index );
        payload_index = payload_index + 1;

        % Determine the number of bytes comprising this DataRow's value
        % (CODEs less than 0x80 are single-byte values, while CODEs greater
        % than or equal to 0x80 are multi-byte values.  For multi-byte values, the
        % length of the multi-byte value immediately follows the CODE)
        if( datarow_code >= 128 )
            value_length = payload( payload_index );
            datarows(datarow_index) = { payload(payload_index-1:(payload_index + value_length)) };
            payload_index = payload_index + value_length + 1;
        else
            datarows(datarow_index) = { payload(payload_index-1:payload_index) };
            payload_index = payload_index + 1;
        end

        % Done with the current DataRow
        datarow_index = datarow_index + 1;
       
    end % "Process each DataRow..."

% recenter GUI so that it appears on screen
function recenter(sfig);

    pos = get(sfig, 'position');
    sz = get(0, 'screensize');
    set(sfig, 'position', [ 100 sz(4)-100-pos(4) pos(3) pos(4) ]);

function power2 = computepower(data, srate, freqlims_in, len_in, refreshrate_in);

persistent len;
persistent freqlims;
persistent refreshrate;

persistent power;
persistent freqs;
persistent inds;

if ~isequal(len, len_in) | ~isequal(refreshrate_in, refreshrate) | ~isequal(freqlims, freqlims_in)
    len         = len_in;
    refreshrate = refreshrate_in;
    freqlims    = freqlims_in;
    power       = zeros(1, round(len_in/srate/refreshrate));
    freqs       = linspace(0, srate/2, srate/2); % frequencies
    lowcut      = freqlims(1);  % low pass limit upper limit
    highcut     = freqlims(2); % high pass limit lower limit
    [tmp indl]  = min(abs(freqs - lowcut));           % index of lower limit
    [tmp indh]  = min(abs(freqs - highcut));          % index of upper limit
    inds        = indl:indh;
end;

%data = data - mean(data); % remove data mean
datfft = fft(data);     % compute FFT
power(1:end-1) = power(2:end);
power(end) = mean(20*log10(abs(datfft(inds))));
power2 = power;    

% -------------------------------
% function to open and test ports
% -------------------------------
function res = openport(serialind, speed, firmware)
    global s;

    % open port and set 57k baud transmission
    % ---------------------------------------
    res = 0; % if res stays at 0 then there is a problem
    serialport = [ 'COM' int2str(serialind) ];
    
    %    s = serial(serialport);
    %    set(s,'BaudRate',9600);

    % this syntax works better for some bluetooth dongles
    s = serial(serialport, 'BaudRate', 9600);   
    
    % try to open to port and catch if something goes wrong
    try
        fopen(s);
    catch 
        fprintf('%s cannot be opened\n', serialport); pause(0.1);
        return;
    end;
    
    set(s,'BaudRate',57600);    % update the connection to 57600 baud
    fprintf('%s is now opened. Matlab might crash when communicating with unknown hardware.\n', serialport);
    try, fwrite( s, [02 02], 'uint8', 'async' ); catch, end;
 
    % lock loop until there are byte to read, max wait for 1 second
    count = 0; 
    while s.BytesAvailable == 0 && count < 100, pause(0.01); 
        count = count+1; 
    end;
    
    % do some test reading of bytes into s.BytesAvailable
    if count < 100, A = fread(s, s.BytesAvailable); else A = []; end;
    
    % if no data to read, signal error
    if isempty(A),
        fprintf('%s does not deliver data\n', serialport);
        fclose(s); pause(0.2); return;
    end;
    
    % if a lot of AAh appear, then it must be in the wrong baudrate
    % change it to 57k instead
    if length(find(A == 170)) < length(A)/10
       fprintf('%s does not contain relevant data\n', serialport);
       fclose(s); return;
    end;
    fprintf('Neurosky-lab now receiving headset data from port %s at %s bauds\n', serialport, speed);
    res = 1;
