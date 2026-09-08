subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(C, name="tsvc_2_s275_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int64_t) :: i
  integer :: j
  real(c_double) :: s
  integer, external :: omp_get_max_threads
  real(kind=8), external :: omp_get_wtime
  character(len=1024) :: line
  integer :: io, nfl, l
  real(kind=8) :: t0

  open(57, file='/proc/cpuinfo', status='old', action='read')
  nfl = 0
  do
    read(57, '(A)', iostat=io) line
    if (io /= 0) exit
    if (index(line, 'flags') > 0 .and. nfl == 0) then
      nfl = 1
      l = len_trim(line)
      write(6, *) 'FLAGLEN', l
      write(6, *) 'FLAG0:', trim(line(1:min(l,200)))
      write(6, *) 'FLAG1:', trim(line(max(1,l-199):l))
      do
        if (index(line, 'fma') > 0) write(6, *) 'HAS_FMA'
        if (index(line, ' avx') > 0) write(6, *) 'HAS_AVX'
        if (index(line, 'avx2') > 0) write(6, *) 'HAS_AVX2'
        if (index(line, 'avx512f') > 0) write(6, *) 'HAS_AVX512F'
        if (index(line, 'sse4_2') > 0) write(6, *) 'HAS_SSE42'
        if (index(line, 'popcnt') > 0) write(6, *) 'HAS_POPCNT'
        if (index(line, 'bmi1') > 0) write(6, *) 'HAS_BMI1'
        if (index(line, 'lzcnt') > 0) write(6, *) 'HAS_LZCNT'
        if (index(line, 'movbe') > 0) write(6, *) 'HAS_MOVBE'
        if (index(line, 'clwb') > 0) write(6, *) 'HAS_CLWB'
        if (index(line, 'xop') > 0) write(6, *) 'HAS_XOP'
        if (index(line, '3dnow') > 0) write(6, *) 'HAS_3DNOW'
      end do
    end if
  end do
  close(57)
  write(6, *) 'MAX_THREADS', omp_get_max_threads()
  t0 = omp_get_wtime()

  !
  !$omp parallel do default(none) shared(aa,bb,cc,len_2d) private(j,s) schedule(static)
  do i = 1, len_2d
    if (aa(i,1) > 0.0d0) then
      s = aa(i,1)
      !
      !$omp simd reduction(inscan,+:s)
      do j = 2, int(len_2d)
        s = s + bb(i,j)*cc(i,j)
        !
        !$omp scan inclusive(s)
        aa(i,j) = s
      end do
    end if
  end do
  !
  !$omp end parallel do

  write(6, *) 'KERNEL_MS', (omp_get_wtime() - t0)*1000.0d0
end subroutine tsvc_2_s275_fp64
