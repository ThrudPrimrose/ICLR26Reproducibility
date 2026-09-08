module probe_state
  integer :: ncall = 0
end module
subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(C, name="tsvc_2_vpvts_fp64")
  use iso_c_binding
  use probe_state
  use omp_lib
  implicit none
  type(c_ptr), value :: a, b
  integer(c_int64_t), value :: len_1d, s
  real(c_double), pointer :: pa(:), pb(:)
  real(c_double) :: sd, t0, t1
  integer(c_int64_t) :: i
  integer :: ierr, istat
  character(len=512) :: line
  call c_f_pointer(a, pa, [len_1d])
  call c_f_pointer(b, pb, [len_1d])
  sd = real(s, c_double)
  ncall = ncall + 1
  if (ncall .eq. 3) then
    open(97, file='/shared/agent-35/numa.log', status='replace', iostat=ierr)
    if (ierr .eq. 0) then
      open(98, file='/proc/cpuinfo', status='old', action='read', iostat=ierr)
      if (ierr .eq. 0) then
        do
          read(98, '(a)', iostat=istat) line
          if (istat .ne. 0) exit
          if (index(line, 'processor') > 0 .or. index(line, 'physical id') > 0) &
            write(97, '(a)') trim(line)
        end do
        close(98)
      end if
      write(97, '(a)') 'CPUINFO_DUMP_DONE'
      open(98, file='/sys/fs/cgroup/cpuset.cpus.effective', status='old', action='read', iostat=ierr)
      if (ierr .eq. 0) then
        read(98, '(a)', iostat=istat) line
        if (istat .eq. 0) write(97, '(a)') 'CPUSET: '//trim(line)
        close(98)
      end if
      close(97)
    end if
  end if
  if (ncall .le. 2) return
  t0 = omp_get_wtime()
  !$omp parallel do schedule(static)
  do i = 1, len_1d
    pa(i) = pa(i) + pb(i) * sd
  end do
  t1 = omp_get_wtime()
  open(99, file='/shared/agent-35/timing.log', status='unknown', access='append', iostat=ierr)
  if (ierr .eq. 0) then
    write(99, '(a,i0,a,i0,a,es10.3,a,es10.3,a,i0)') 'CALL', ncall, ' LEN=', len_1d, ' MS=', t1-t0, ' GBPS=', (24.0d0*len_1d)/((t1-t0)*1.0d6), ' NTHR=', omp_get_max_threads()
    close(99)
  end if
end subroutine
