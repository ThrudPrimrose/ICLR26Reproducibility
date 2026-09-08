subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name="tsvc_2_s1232_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout), dimension(*) :: aa
  real(c_double), intent(in), dimension(*) :: bb, cc
  integer(c_int64_t), value, intent(in) :: LEN_2D, VLEN
  integer(c_int64_t) :: l, v, i, m, r, t, nt, n, mcap
  l = LEN_2D
  v = VLEN
  if (l <= 0) return
  if (v < 0) v = 0
  ! total element count: sum over j of max(0, l - v*j)
  n = 0
  if (v > 0) then
     mcap = (l - 1) / v + 1
     if (mcap > l) mcap = l
     n = mcap * l - v * mcap * (mcap - 1) / 2
  else
     n = l * l
  end if
  if (n < 131072) then
     ! small: serial
     if (v > 0) then
        do i = 0, l-1
           m = i / v
           if (m > l-1) m = l-1
           do r = 0, m
              aa(i*l+1+r) = bb(i*l+1+r) + cc(i*l+1+r)
           end do
        end do
     else
        do i = 0, l-1
           do r = 0, l-1
              aa(i*l+1+r) = bb(i*l+1+r) + cc(i*l+1+r)
           end do
        end do
     end if
  else
     !$omp parallel default(none) shared(aa,bb,cc,l,v) private(i,m,r,t,nt)
     nt = omp_get_num_threads()
     t = omp_get_thread_num()
     if (v > 0) then
        do i = t, l-1, nt
           m = i / v
           if (m > l-1) m = l-1
           do r = 0, m
              aa(i*l+1+r) = bb(i*l+1+r) + cc(i*l+1+r)
           end do
        end do
     else
        do i = t, l-1, nt
           do r = 0, l-1
              aa(i*l+1+r) = bb(i*l+1+r) + cc(i*l+1+r)
           end do
        end do
     end if
     !$omp end parallel
  end if
end subroutine tsvc_2_s1232_fp64
